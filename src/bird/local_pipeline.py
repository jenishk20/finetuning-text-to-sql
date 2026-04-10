"""
Run a local Qwen model against the BIRD train set for delta learning.

Loads a small Qwen model (3B or 1.5B) in 4-bit NF4 and runs it over the
BIRD train set with SQL execution to determine correctness. The output
is fed into build_bird_delta_pairs.py to create preference pairs for DPO.

Run BOTH models, then build pairs:
    python src/bird/local_pipeline.py --model Qwen/Qwen2.5-Coder-3B-Instruct
    python src/bird/local_pipeline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct
    python src/bird/build_bird_delta_pairs.py --qwen3b results/bird_local_Qwen_Qwen2.5-Coder-3B-Instruct_XXXX.json \\
                                               --qwen1b results/bird_local_Qwen_Qwen2.5-Coder-1.5B-Instruct_XXXX.json

Defaults assume this EC2 layout (g5.xlarge, Deep Learning AMI):
    ~/bird_train.json              BIRD train questions (9,428 entries)
    ~/bird_train_databases/        train SQLite databases (~30GB)
    ~/results/                     output directory

Override any path with CLI args if your layout differs.

Requirements (already on the EC2 Deep Learning AMI):
    pip install transformers peft bitsandbytes accelerate
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.evaluator import compare_results, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# EC2 DEFAULT PATHS  (override with CLI args if needed)
# ─────────────────────────────────────────────────────────────────────────────
HOME = Path.home()
DEFAULT_TRAIN_JSON = HOME / "bird_train.json"
DEFAULT_DB_DIR     = HOME / "bird_train_databases"
DEFAULT_OUTPUT_DIR = HOME / "results"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE — must match build_bird_delta_pairs.py and DPO training
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def build_instruction(question: str, schema: str, evidence: str = "") -> str:
    """Build the prompt used both for inference and stored in the output for
    build_bird_delta_pairs.py to reuse."""
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load a Qwen model in 4-bit NF4 (same config as SFT training)."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return model, tokenizer


def generate_sql(model, tokenizer, schema: str, question: str, evidence: str) -> str:
    """Generate SQL using the local model.

    Uses build_instruction() so the prompt format exactly matches what is stored
    in the output JSON and used for DPO training — no distribution shift.
    """
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

    # Strip markdown fences
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fence_match:
        response = fence_match.group(1).strip()

    # Ensure trailing semicolon
    if not response.endswith(";"):
        response += ";"

    return response


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def get_train_db_path(db_id: str, db_dir: Path) -> str:
    """Resolve train database path. Structure mirrors dev_databases/."""
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if db_path.exists():
        return str(db_path)

    # Fallback: search for any .sqlite in the db_id directory
    db_subdir = db_dir / db_id
    if db_subdir.exists():
        for f in db_subdir.glob("*.sqlite"):
            return str(f)

    raise FileNotFoundError(
        f"No SQLite database found for db_id={db_id} in {db_dir}\n"
        f"Download BIRD train databases and place at {db_dir}/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA LOADING (inline — avoids project root dependency on EC2)
# ─────────────────────────────────────────────────────────────────────────────

def get_schema_from_sqlite(db_path: str) -> str:
    """Extract CREATE TABLE statements from a SQLite database."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    tables = cursor.fetchall()
    conn.close()
    return "\n\n".join(sql for _, sql in tables if sql)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_local_pipeline(
    model_id: str,
    train_json: Path,
    db_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    resume_file: str | None = None,
):
    model_label = model_id.replace("/", "_").replace(".", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load train questions ──────────────────────────────────────────────────
    print(f"\nLoading train questions from {train_json}...")
    with open(train_json) as f:
        train_questions = json.load(f)

    # Normalize: ensure each question has a question_id
    # BIRD train.json uses integer question_id; fall back to enumerate index
    for i, q in enumerate(train_questions):
        if "question_id" not in q:
            q["question_id"] = i

    if limit:
        train_questions = train_questions[:limit]

    # ── Resume support ────────────────────────────────────────────────────────
    completed_ids: set = set()
    existing_results: list[dict] = []
    if resume_file and Path(resume_file).exists():
        with open(resume_file) as f:
            saved = json.load(f)
            existing_results = saved.get("results", [])
            completed_ids = {r["question_id"] for r in existing_results}
        print(f"Resuming — {len(completed_ids)} questions already done")

    total_to_run = len(train_questions) - len(completed_ids)

    print(f"\n{'=' * 70}")
    print(f"  BIRD LOCAL PIPELINE — Delta Learning")
    print(f"  Model:     {model_id}")
    print(f"  Questions: {len(train_questions)} (train set)")
    print(f"  To run:    {total_to_run}")
    print(f"  DB dir:    {db_dir}")
    print(f"{'=' * 70}\n")

    if not db_dir.exists():
        print(f"ERROR: Train database directory not found: {db_dir}")
        print(f"Download BIRD train databases and place at {db_dir}/")
        print(f"Structure: {db_dir}/{{db_id}}/{{db_id}}.sqlite")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(model_id)

    # ── Run ───────────────────────────────────────────────────────────────────
    schema_cache: dict[str, str] = {}
    results = list(existing_results)
    run_count = 0

    for q in train_questions:
        question_id = q["question_id"]
        if question_id in completed_ids:
            continue

        run_count += 1
        db_id      = q["db_id"]
        question   = q["question"]
        evidence   = q.get("evidence", "").strip()
        gold_sql   = q.get("SQL") or q.get("query", "")
        difficulty = q.get("difficulty", "unknown")

        print(f"  [{run_count}/{total_to_run}] Q{question_id} | db={db_id} | {difficulty}")

        # Load schema
        if db_id not in schema_cache:
            try:
                db_path = get_train_db_path(db_id, db_dir)
                schema_cache[db_id] = get_schema_from_sqlite(db_path)
            except FileNotFoundError as e:
                print(f"    SKIP — {e}")
                continue

        schema  = schema_cache[db_id]
        db_path = get_train_db_path(db_id, db_dir)

        # Generate SQL
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
                "difficulty": difficulty,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "generation_error": str(e),
                "instruction": build_instruction(question, schema, evidence),
                "eval": {
                    "generated_executed": False,
                    "gold_executed": False,
                    "result_match": False,
                    "row_count_match": False,
                    "generated_row_count": 0,
                    "gold_row_count": 0,
                    "column_match": False,
                    "details": f"Generation error: {e}",
                },
            })
            continue

        # Execute and evaluate
        gen_exec  = execute_sqlite_query(generated_sql, db_path)
        gold_exec = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(gen_exec, gold_exec)

        status   = "PASS" if evaluation["result_match"] else "FAIL"
        exec_ok  = "OK"   if gen_exec["success"] else "ERR"
        print(f"    Exec: {exec_ok} | {status} | {evaluation['details'][:70]}")

        results.append({
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "generation_time_s": gen_time,
            "instruction": build_instruction(question, schema, evidence),
            "gen_execution": {
                "success": gen_exec["success"],
                "row_count": gen_exec["row_count"],
                "execution_time_ms": gen_exec["execution_time_ms"],
                "error": gen_exec["error"],
                "columns": gen_exec["columns"],
            },
            "gold_execution": {
                "success": gold_exec["success"],
                "row_count": gold_exec["row_count"],
                "execution_time_ms": gold_exec["execution_time_ms"],
                "error": gold_exec["error"],
                "columns": gold_exec["columns"],
            },
            "eval": evaluation,
        })

        # Checkpoint every 100 questions
        if run_count % 100 == 0:
            output_file = _save(model_id, model_label, results, output_dir)
            print(f"    --- Checkpoint saved ({run_count} done) → {output_file.name} ---")

    # ── Final save + metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(results)
    output_file = _save(model_id, model_label, results, output_dir, metrics)

    correct = sum(1 for r in results if r["eval"]["result_match"])
    total   = len(results)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {model_id}")
    print(f"{'=' * 70}")
    print(f"  Total questions:    {total}")
    print(f"  Correct (pass):     {correct}  ({correct/total:.1%})" if total else "  No results")
    print(f"  Result accuracy:    {metrics.get('result_accuracy', 0):.1%}")

    # Breakdown by difficulty
    for diff in ["simple", "moderate", "challenging"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r["eval"]["result_match"])
            print(f"  {diff.capitalize():<12}: {diff_correct}/{len(diff_results)} ({diff_correct/len(diff_results):.1%})")

    print(f"{'=' * 70}")
    print(f"  Results saved to: {output_file}\n")
    print(f"  Next: run same command with the other model, then:")
    print(f"  python src/bird/build_bird_delta_pairs.py --qwen3b <3b_file> --qwen1b <1b_file>\n")

    return output_file


def _save(model_id: str, model_label: str, results: list, output_dir: Path, metrics: dict | None = None) -> Path:
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bird_local_{model_label}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model": model_id,
                "timestamp": timestamp,
                "benchmark": "bird_train",
                "total_completed": len(results),
            },
            "metrics": metrics or {},
            "results": results,
        }, f, indent=2, default=str)

    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local Qwen model on BIRD train set for delta learning"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-3B-Instruct)"
    )
    parser.add_argument(
        "--train-json", type=Path, default=DEFAULT_TRAIN_JSON,
        help=f"Path to bird_train.json (default: {DEFAULT_TRAIN_JSON})"
    )
    parser.add_argument(
        "--db-dir", type=Path, default=DEFAULT_DB_DIR,
        help=f"Path to BIRD train databases directory (default: {DEFAULT_DB_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run first N questions (useful for smoke tests)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a previous results file to resume from"
    )
    args = parser.parse_args()

    run_local_pipeline(
        model_id=args.model,
        train_json=args.train_json,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        resume_file=args.resume,
    )
