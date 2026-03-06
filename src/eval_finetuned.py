"""
Evaluate a fine-tuned model on the Spider benchmark.

Loads the QLoRA adapter on top of the base model and runs the same
Spider dev evaluation as spider_pipeline.py, but using local inference
instead of the Grok API.

Usage:
    python -m src.eval_finetuned                        # all 1034 questions
    python -m src.eval_finetuned --limit 50             # first 50 only
    python -m src.eval_finetuned --base_only            # eval base model WITHOUT fine-tuning
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

from src.sqlite_executor import execute_sqlite_query
from src.schema_loader import get_schema_from_sqlite, get_db_path
from src.evaluator import compare_results, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DATA_DIR = PROJECT_ROOT / "data" / "spider_data"
LORA_DIR = PROJECT_ROOT / "models" / "qwen-7b-sql-lora"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def load_model(use_lora: bool = True):
    """Load the base model (optionally with LoRA adapter) in 4-bit."""
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:
        if not LORA_DIR.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {LORA_DIR}. Run src.finetune first."
            )
        print(f"Loading LoRA adapter from {LORA_DIR}...")
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, schema: str, question: str) -> str:
    """Generate SQL using the local model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"### Database Schema:\n{schema}\n\n"
                f"### Question:\n{question}\n\n"
                f"### SQL Query:"
            ),
        },
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

    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", response, re.DOTALL)
    if fence_match:
        response = fence_match.group(1).strip()

    if response.endswith(";"):
        return response
    return response + ";"


def run_evaluation(limit: int | None = None, base_only: bool = False):
    use_lora = not base_only
    model_label = "qwen-7b-sql-lora" if use_lora else "qwen-7b-base"

    model, tokenizer = load_model(use_lora=use_lora)

    with open(SPIDER_DATA_DIR / "dev.json") as f:
        dev_questions = json.load(f)

    if limit:
        dev_questions = dev_questions[:limit]

    print(f"\n{'=' * 70}")
    print(f"  SPIDER EVALUATION — {model_label}")
    print(f"  Questions: {len(dev_questions)}")
    print(f"  LoRA adapter: {'Yes' if use_lora else 'No (base model only)'}")
    print(f"{'=' * 70}\n")

    schema_cache: dict[str, str] = {}
    results = []

    for idx, q in enumerate(dev_questions):
        db_id = q["db_id"]
        question = q["question"]
        gold_sql = q["query"]

        print(f"  [{idx + 1}/{len(dev_questions)}] Q{idx} | db={db_id}")

        if db_id not in schema_cache:
            try:
                db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)
                schema_cache[db_id] = get_schema_from_sqlite(db_path)
            except FileNotFoundError as e:
                print(f"    SKIP — {e}")
                continue
        schema = schema_cache[db_id]
        db_path = get_db_path(str(SPIDER_DATA_DIR), db_id)

        try:
            start = time.time()
            generated_sql = generate_sql(model, tokenizer, schema, question)
            gen_time = round(time.time() - start, 2)
            print(f"    Generated ({gen_time}s): {generated_sql[:100]}...")
        except Exception as e:
            print(f"    GENERATION ERROR: {e}")
            results.append({
                "spider_index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "error": str(e),
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

        gen_exec = execute_sqlite_query(generated_sql, db_path)
        gold_exec = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(gen_exec, gold_exec)

        status = "PASS" if evaluation["result_match"] else "FAIL"
        exec_ok = "OK" if gen_exec["success"] else "ERR"
        print(f"    Exec: {exec_ok} | {status} | {evaluation['details'][:80]}")

        results.append({
            "spider_index": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "generation_time_s": gen_time,
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
            _save(model_label, results, len(dev_questions))
            print(f"    --- Checkpoint ({idx + 1} done) ---")

    metrics = compute_metrics(results)
    output_file = _save(model_label, results, len(dev_questions), metrics)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {model_label}")
    print(f"{'=' * 70}")
    print(f"  Total questions:      {metrics.get('total_questions', 0)}")
    print(f"  Execution accuracy:   {metrics.get('execution_accuracy', 0):.1%}")
    print(f"  Result accuracy:      {metrics.get('result_accuracy', 0):.1%}")
    print(f"  Row count accuracy:   {metrics.get('row_count_accuracy', 0):.1%}")
    print(f"{'=' * 70}")
    print(f"  Results saved to: {output_file}\n")

    return metrics


def _save(model_label, results, total_questions, metrics=None):
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_label}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model": model_label,
                "base_model": BASE_MODEL,
                "timestamp": timestamp,
                "benchmark": "spider_dev",
                "total_questions": total_questions,
                "completed": len(results),
            },
            "metrics": metrics or {},
            "results": results,
        }, f, indent=2, default=str)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base_only", action="store_true", help="Evaluate base model without LoRA")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, base_only=args.base_only)
