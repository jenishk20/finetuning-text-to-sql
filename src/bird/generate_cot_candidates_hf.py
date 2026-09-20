"""
Generate K chain-of-thought candidates per BIRD dev question using plain
HuggingFace transformers (NO vLLM).

Why this exists: the vLLM env on /scratch was purged, and reinstalling vLLM
pulls a brittle dependency chain (mistral-common, etc.). Generation only needs
torch + transformers, which we can restore trivially. This loads the LOCAL
merged model, so there is no HuggingFace download and no proxy dependency.

Output format matches exactly what score_pass_k.py expects (the same schema
build_self_sampling_pairs --save-candidates produced): a JSON list where each
record has question_id, db_id, gold_sql, db_path, difficulty, candidates[K].

Usage:
    python -m src.bird.generate_cot_candidates_hf \
        --model    /scratch/phalle.y/bird_cot_sft_merged_7b \
        --dev-json /scratch/phalle.y/bird_dev/dev_20240627/dev.json \
        --db-dir   /scratch/phalle.y/bird_dev/dev_20240627/dev_databases \
        --out      /scratch/phalle.y/results_cot_passk/dev_cot_candidates_k8.json \
        --k 8 --temperature 0.8 --max-new-tokens 768 [--limit 200]

Must run on a GPU node. Writes incrementally after every batch, so a crash
keeps completed work (delete the --out file to force a clean re-run).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# These imports pull only `re`/`typing` at module load (the vLLM engine is
# imported lazily inside its own class), so this works in a vLLM-free env.
from src.bird.inference import build_cot_instruction, COT_SYSTEM_PROMPT


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    parts = [r[0] for r in cur.fetchall() if r[0]]
    conn.close()
    return "\n\n".join(parts)


def resolve_db(db_dir: Path, db_id: str) -> str | None:
    p = Path(db_dir) / db_id / f"{db_id}.sqlite"
    return str(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Generate K CoT candidates on BIRD dev (HF transformers)")
    ap.add_argument("--model",          type=str,  required=True, help="Local merged model dir or HF id")
    ap.add_argument("--dev-json",       type=Path, required=True)
    ap.add_argument("--db-dir",         type=Path, required=True)
    ap.add_argument("--out",            type=Path, required=True)
    ap.add_argument("--k",              type=int,   default=8, help="samples per question")
    ap.add_argument("--temperature",    type=float, default=0.8)
    ap.add_argument("--top-p",          type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int,   default=768)
    ap.add_argument("--limit",          type=int,   default=0, help="0 = all dev questions")
    ap.add_argument("--batch-size",     type=int,   default=4,
                    help="prompts per forward pass (each expands to k sequences). Lower if OOM.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible — run this on a GPU node.")

    dev = json.load(open(args.dev_json))
    if args.limit:
        dev = dev[: args.limit]

    print(f"Loading tokenizer + model from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    # ── Build prompts + carry-through metadata ────────────────────────────────
    schema_cache: dict[str, str] = {}
    recs = []
    skipped = 0
    for i, q in enumerate(dev):
        db_id = q["db_id"]
        db_path = resolve_db(args.db_dir, db_id)
        if db_path is None:
            skipped += 1
            continue
        if db_id not in schema_cache:
            schema_cache[db_id] = get_schema(db_path)
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user",   "content": build_cot_instruction(
                q["question"], schema_cache[db_id], q.get("evidence", "").strip())},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        recs.append({
            "question_id": q.get("question_id", i),
            "db_id":       db_id,
            "gold_sql":    q.get("SQL") or q.get("query", ""),
            "db_path":     db_path,
            "difficulty":  q.get("difficulty", "unknown"),
            "prompt":      prompt,
        })
    print(f"Prepared {len(recs)} questions ({skipped} skipped for missing db). "
          f"Generating k={args.k} at temp={args.temperature} ...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_records = []
    start = time.time()
    for b in range(0, len(recs), args.batch_size):
        batch = recs[b: b + args.batch_size]
        enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id,
            )
        # generate returns (len(batch)*k, seq); strip the prompt, decode, regroup.
        gen = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for j, r in enumerate(batch):
            rec = {k: v for k, v in r.items() if k != "prompt"}
            rec["candidates"] = texts[j * args.k:(j + 1) * args.k]
            out_records.append(rec)

        done = min(b + args.batch_size, len(recs))
        rate = (time.time() - start) / 60
        print(f"  [{done}/{len(recs)}]  {rate:.1f} min elapsed", flush=True)
        json.dump(out_records, open(args.out, "w"), indent=2)  # incremental save

    json.dump(out_records, open(args.out, "w"), indent=2)
    print(f"Done. Saved {len(out_records)} candidate sets to {args.out}")


if __name__ == "__main__":
    main()
