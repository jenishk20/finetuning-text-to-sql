"""
pass@K diagnostic for a Spider SFT/DPO adapter — measures on-policy headroom.

For each dev question: GREEDY decode (the deployable number) + K sampled completions,
execute all against the gold DB, and report:
  - greedy accuracy        (≈ your eval number, ~77%)
  - pass@K                 (fraction where ANY of the K samples is correct) = on-policy ceiling
  - learnable pool         (questions with a MIX of correct+wrong across greedy+K) = GRPO/SS-DPO signal
  - unrecoverable          (0 correct in greedy+K) = out of reach without a stronger base

Decision:
  pass@K >> greedy  → real headroom; GRPO can lift greedy toward pass@K → worth training.
  pass@K ~ greedy   → model is near its ceiling; RL won't move it → 77% is your result.

Usage (GPU):
    python -m src.spider.passk_diagnostic \
        --adapter  /scratch/phalle.y/spider_sft_adapter_7b/final_adapter \
        --data-dir /home/phalle.y/Jenish-DPO-GRPO \
        --limit 300 --k 8
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.schema_loader import get_schema_from_sqlite, get_db_path
from src.shared.evaluator import compare_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Same prompt as src/spider/eval_finetuned.py so greedy here matches your 77% eval.
SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def load_model(adapter: Path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def _clean(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    text = text.strip()
    return text if text.endswith(";") else text + ";"


def _prompt(tokenizer, schema: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### Database Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL Query:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, schema, question, k, temperature, top_p, max_new_tokens):
    text = _prompt(tokenizer, schema, question)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    ilen = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        greedy = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        sampled = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=top_p, num_return_sequences=k,
                                 pad_token_id=tokenizer.eos_token_id)
    greedy_sql = _clean(tokenizer.decode(greedy[0][ilen:], skip_special_tokens=True))
    samples = [_clean(tokenizer.decode(seq[ilen:], skip_special_tokens=True)) for seq in sampled]
    return greedy_sql, samples


def main():
    ap = argparse.ArgumentParser(description="pass@K headroom diagnostic for a Spider adapter")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "spider_data")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "spider_passk")
    ap.add_argument("--limit", type=int, default=300, help="dev questions to sample (300 ≈ enough signal)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tokenizer = load_model(args.adapter)
    with open(args.data_dir / "dev.json") as f:
        dev = json.load(f)
    if args.limit:
        dev = dev[:args.limit]

    schema_cache: dict[str, str] = {}
    greedy_ok_n = passk_n = learnable_n = unrecoverable_n = 0
    per_q = []

    print(f"pass@{args.k} diagnostic on {len(dev)} Spider dev questions (T={args.temperature})\n")
    for idx, q in enumerate(dev):
        db_id, question, gold = q["db_id"], q["question"], q["query"]
        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = get_schema_from_sqlite(get_db_path(str(args.data_dir), db_id))
            except FileNotFoundError:
                continue
        db_path = get_db_path(str(args.data_dir), db_id)
        gold_exec = execute_sqlite_query(gold, db_path)

        greedy_sql, samples = generate(model, tokenizer, schema_cache[db_id], question,
                                       args.k, args.temperature, args.top_p, args.max_new_tokens)
        greedy_ok = compare_results(execute_sqlite_query(greedy_sql, db_path), gold_exec)["result_match"]
        sample_oks = [compare_results(execute_sqlite_query(s, db_path), gold_exec)["result_match"] for s in samples]

        pool = [greedy_ok] + sample_oks
        n_correct = sum(pool)
        any_correct = n_correct > 0
        all_correct = n_correct == len(pool)

        greedy_ok_n += int(greedy_ok)
        passk_n += int(any(sample_oks))          # pass@K over the K SAMPLED completions
        if 0 < n_correct < len(pool):
            learnable_n += 1                      # mix of right+wrong → yields on-policy pairs/signal
        if not any_correct:
            unrecoverable_n += 1
        per_q.append({"db_id": db_id, "greedy_ok": greedy_ok, "k_correct": sum(sample_oks), "k": args.k})

        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(dev)}] greedy={greedy_ok_n/(idx+1):.1%}  "
                  f"pass@{args.k}={passk_n/(idx+1):.1%}  learnable={learnable_n/(idx+1):.1%}")

    n = len(per_q) or 1
    summary = {
        "adapter": str(args.adapter), "questions": len(per_q), "k": args.k, "temperature": args.temperature,
        "greedy_accuracy": round(greedy_ok_n / n, 4),
        "pass_at_k": round(passk_n / n, 4),
        "learnable_pool": round(learnable_n / n, 4),
        "unrecoverable": round(unrecoverable_n / n, 4),
        "headroom_pp": round((passk_n - greedy_ok_n) / n * 100, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"passk_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_question": per_q}, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  pass@{args.k} DIAGNOSTIC — {len(per_q)} questions")
    print(f"{'='*66}")
    print(f"  greedy accuracy:   {summary['greedy_accuracy']:.1%}")
    print(f"  pass@{args.k}:            {summary['pass_at_k']:.1%}")
    print(f"  HEADROOM:          +{summary['headroom_pp']}pp   (pass@K − greedy)")
    print(f"  learnable pool:    {summary['learnable_pool']:.1%}  (questions RL/DPO can act on)")
    print(f"  unrecoverable:     {summary['unrecoverable']:.1%}  (0/{args.k+1} correct — need stronger base)")
    print(f"{'='*66}")
    print(f"  → headroom ≥ ~6pp: GRPO worth running.  ≈0: 77% is near the ceiling.")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
