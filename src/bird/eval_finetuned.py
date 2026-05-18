"""
Evaluate a fine-tuned model on the BIRD benchmark dev set.

Loads the DPO adapter on top of the base model and runs BIRD dev evaluation
using local inference. Prompt format matches exactly what was used during
DPO training (including evidence field).

Usage:
    # Full eval with DPO adapter
    python -m src.bird.eval_finetuned \
        --adapter  /scratch/phalle.y/bird_frontier_dpo_adapter/final_adapter \
        --dev-json /scratch/phalle.y/bird_dev/dev.json \
        --db-dir   /scratch/phalle.y/bird_dev/dev_databases \
        --output-dir /scratch/phalle.y/results

    # Quick smoke test (first 50 questions)
    python -m src.bird.eval_finetuned ... --limit 50

    # Base model only (no adapter — for comparison)
    python -m src.bird.eval_finetuned ... --base-only

    # Resume an interrupted run
    python -m src.bird.eval_finetuned ... --resume /scratch/phalle.y/results/bird_eval_xxx.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results, compute_metrics
from src.bird.inference import build_instruction, SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(adapter: Path | None, base_only: bool = False, base_model: str = DEFAULT_BASE_MODEL):
    """Load base model in 4-bit NF4, optionally with DPO LoRA adapter."""
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {base_model} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if not base_only:
        if adapter and Path(adapter).exists():
            print(f"Loading DPO adapter from {adapter}...")
            model = PeftModel.from_pretrained(model, str(adapter))
            print("Adapter loaded.")
        else:
            print(f"WARNING: Adapter not found at {adapter} — evaluating base model only")

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(model, tokenizer, schema: str, question: str, evidence: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_instruction(question, schema, evidence)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fence_match:
        response = fence_match.group(1).strip()

    if not response.endswith(";"):
        response += ";"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS (inlined — avoids path issues on HPC)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_path(db_dir: Path, db_id: str) -> str:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)
    subdir = db_dir / db_id
    if subdir.exists():
        for f in subdir.glob("*.sqlite"):
            return str(f)
    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {db_dir}")


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "\n\n".join(tables)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    dev_json: Path,
    db_dir: Path,
    adapter: Path | None,
    output_dir: Path,
    limit: int | None = None,
    base_only: bool = False,
    resume_file: Path | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
):
    short_name  = base_model.split("/")[-1].lower().replace("qwen2.5-coder-", "qwen").replace("-instruct", "")
    model_label = f"{short_name}-bird-base" if base_only else f"{short_name}-bird-dpo"

    # ── Resume support ────────────────────────────────────────────────────────
    completed_ids: set = set()
    existing_results: list = []
    if resume_file and Path(resume_file).exists():
        with open(resume_file) as f:
            saved = json.load(f)
            existing_results = saved.get("results", [])
            completed_ids = {str(r["question_id"]) for r in existing_results}
        print(f"Resuming — {len(completed_ids)} questions already done")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(dev_json) as f:
        dev_questions = json.load(f)

    if limit:
        dev_questions = dev_questions[:limit]

    total  = len(dev_questions)
    to_run = total - len(completed_ids)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(adapter, base_only, base_model=base_model)

    print(f"\n{'=' * 70}")
    print(f"  BIRD EVALUATION — {model_label}")
    print(f"  Questions:  {total} total | {to_run} to run")
    print(f"  Dev JSON:   {dev_json}")
    print(f"  DB dir:     {db_dir}")
    print(f"  Adapter:    {adapter or 'none (base model)'}")
    print(f"{'=' * 70}\n")

    schema_cache: dict = {}
    results = list(existing_results)

    for idx, q in enumerate(dev_questions):
        question_id = str(q.get("question_id", idx))
        if question_id in completed_ids:
            continue

        db_id    = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "").strip()
        gold_sql = q.get("SQL") or q.get("query", "")

        print(f"  [{idx + 1}/{total}] Q{question_id} | db={db_id}")

        if db_id not in schema_cache:
            try:
                db_path = get_db_path(db_dir, db_id)
                schema_cache[db_id] = (get_schema(db_path), db_path)
            except FileNotFoundError as e:
                print(f"    SKIP — {e}")
                continue

        schema, db_path = schema_cache[db_id]

        try:
            start = time.time()
            generated_sql = generate_sql(model, tokenizer, schema, question, evidence)
            gen_time = round(time.time() - start, 2)
            print(f"    Generated ({gen_time}s): {generated_sql[:100]}")
        except Exception as e:
            print(f"    GENERATION ERROR: {e}")
            results.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "difficulty": q.get("difficulty", "unknown"),
                "error": str(e),
                "eval": {"result_match": False, "details": f"Generation error: {e}"},
            })
            continue

        gen_exec   = execute_sqlite_query(generated_sql, db_path)
        gold_exec  = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(gen_exec, gold_exec)

        status = "PASS" if evaluation["result_match"] else "FAIL"
        exec_ok = "OK"  if gen_exec["success"]      else "ERR"
        print(f"    Exec: {exec_ok} | {status} | {evaluation['details'][:80]}")

        results.append({
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "generation_time_s": gen_time,
            "difficulty": q.get("difficulty", "unknown"),
            "gen_execution": {
                "success": gen_exec["success"],
                "row_count": gen_exec["row_count"],
                "error": gen_exec["error"],
            },
            "gold_execution": {
                "success": gold_exec["success"],
                "row_count": gold_exec["row_count"],
                "error": gold_exec["error"],
            },
            "eval": evaluation,
        })

        if (idx + 1) % 50 == 0:
            _save(model_label, results, total, output_dir)
            print(f"    --- Checkpoint ({idx + 1}/{total} done) ---")

    metrics = compute_metrics(results)
    output_file = _save(model_label, results, total, output_dir, metrics, base_model=base_model)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {model_label}")
    print(f"{'=' * 70}")
    print(f"  Total questions:    {metrics.get('total_questions', 0)}")
    print(f"  Execution accuracy: {metrics.get('execution_accuracy', 0):.1%}")
    print(f"  Result accuracy:    {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Row count accuracy: {metrics.get('row_count_accuracy', 0):.1%}")
    print(f"{'=' * 70}")
    print(f"  Results saved to: {output_file}\n")

    return metrics


def _save(model_label, results, total, output_dir, metrics=None, base_model=DEFAULT_BASE_MODEL):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bird_eval_{model_label}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model":           model_label,
                "base_model":      base_model,
                "timestamp":       timestamp,
                "benchmark":       "bird_dev",
                "total_questions": total,
                "completed":       len(results),
            },
            "metrics": metrics or {},
            "results": results,
        }, f, indent=2, default=str)
    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on BIRD dev set")

    parser.add_argument("--adapter",    type=Path, default=None,
                        help="Path to DPO LoRA adapter directory (final_adapter/)")
    parser.add_argument("--dev-json",   type=Path,
                        default=PROJECT_ROOT / "data" / "bird_data" / "dev.json",
                        help="Path to BIRD dev.json")
    parser.add_argument("--db-dir",     type=Path,
                        default=PROJECT_ROOT / "data" / "bird_data" / "dev_databases",
                        help="Path to BIRD dev_databases/ directory")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results",
                        help="Directory to save results JSON")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Evaluate first N questions only (smoke test)")
    parser.add_argument("--base-only",  action="store_true",
                        help="Evaluate base model without adapter (baseline comparison)")
    parser.add_argument("--resume",     type=Path, default=None,
                        help="Path to partial results JSON to resume an interrupted run")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"HF base model ID (default: {DEFAULT_BASE_MODEL}). Use Qwen/Qwen2.5-Coder-14B-Instruct for 14B.")

    args = parser.parse_args()

    run_evaluation(
        dev_json=args.dev_json,
        db_dir=args.db_dir,
        adapter=args.adapter,
        output_dir=args.output_dir,
        limit=args.limit,
        base_only=args.base_only,
        resume_file=args.resume,
        base_model=args.base_model,
    )
