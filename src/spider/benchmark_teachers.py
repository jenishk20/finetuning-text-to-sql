"""
Benchmark candidate DPO-teacher models on the Spider dev set via W&B Inference.

Runs one or more models over the Spider dev set (default: all 1,034 questions),
executes each generated SQL against the gold SQLite database, compares result sets
to the gold query's results, and reports result / execution accuracy per model.
Prints a final ranked summary across all models and writes a combined summary JSON.

WHY THIS IS NOT CONTAMINATION
─────────────────────────────
This runs on the DEV set, which is the evaluation split. That is fine HERE because
we are only *measuring* teacher accuracy (selection) and recording clean frontier
*baselines*. We do NOT train on dev. The DPO preference pairs are built from the
TRAIN set in a separate step, so nothing measured here leaks into training.

Usage:
    # All four default candidates, full dev set
    python -m src.spider.benchmark_teachers

    # A couple of models, quick smoke test
    python -m src.spider.benchmark_teachers \
        --models qwen3-coder-480b deepseek-v4-pro --limit 25

    # Resume a model whose run was interrupted
    python -m src.spider.benchmark_teachers --models deepseek-v4-pro \
        --resume results/teacher_bakeoff/spider_bakeoff_deepseek-ai_DeepSeek-V4-Pro_XXXX.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.shared.llm_client import call_llm, resolve_model, WANDB_MODELS
from src.shared.sqlite_executor import execute_sqlite_query
from src.shared.schema_loader import get_schema_from_sqlite, get_db_path
from src.shared.evaluator import compare_results, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default DPO-teacher candidates (shortcuts resolve via WANDB_MODELS).
# Qwen3-Coder-480B is the locked teacher A; the DeepSeek pair decides teacher B;
# GLM-5.2 is an extra baseline for the comparison table.
DEFAULT_MODELS = [
    "qwen3-coder-480b",
    "deepseek-v4-pro",
    "deepseek-v3.1",
    "glm-5.2",
]


def model_label(full_model: str) -> str:
    """Filesystem-safe label, matching the existing spider_bakeoff_* naming."""
    return full_model.replace("/", "_").replace(".", "-")


def load_dev(data_dir: Path) -> list[dict]:
    with open(data_dir / "dev.json") as f:
        return json.load(f)


def call_with_retry(retries: int, backoff: float, **kwargs) -> dict:
    """call_llm with exponential backoff — W&B Inference can rate-limit / 5xx."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return call_llm(**kwargs)
        except Exception as e:  # noqa: BLE001 - surface any provider error, then retry
            last_err = e
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"      retry {attempt + 1}/{retries} after error: "
                      f"{str(e)[:140]} (sleep {wait:.0f}s)")
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def _failed_eval(detail: str) -> dict:
    return {
        "generated_executed": False, "gold_executed": False,
        "result_match": False, "row_count_match": False,
        "generated_row_count": 0, "gold_row_count": 0,
        "column_match": False, "details": detail,
    }


def _save_model_run(provider, full_model, label, results, total_tokens,
                    total_questions, output_dir, metrics=None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"spider_bakeoff_{label}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({
            "metadata": {
                "provider": provider,
                "model": full_model,
                "timestamp": timestamp,
                "benchmark": "spider_dev",
                "purpose": "teacher_selection_and_baseline",
                "total_questions": total_questions,
                "completed": len(results),
                "total_tokens": total_tokens,
            },
            "metrics": metrics or {},
            "results": results,
        }, f, indent=2, default=str)
    return out_file


def run_one_model(model, provider, dev, data_dir, output_dir,
                  temperature, max_tokens, retries, backoff, resume_file):
    full_model = resolve_model(model, provider)
    label = model_label(full_model)

    completed: dict[int, dict] = {}
    if resume_file and Path(resume_file).exists():
        with open(resume_file) as f:
            saved = json.load(f)
        completed = {r["spider_index"]: r for r in saved.get("results", [])}
        print(f"  Resuming {label} from {resume_file} — {len(completed)} already done")

    print(f"\n{'=' * 84}")
    print(f"  TEACHER: {full_model}   ({provider})")
    print(f"  Questions: {len(dev)} (Spider dev) | already done: {len(completed)}")
    print(f"{'=' * 84}")

    schema_cache: dict[str, str] = {}
    results = list(completed.values())
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    run_count = 0
    t0 = time.time()

    for idx, q in enumerate(dev):
        if idx in completed:
            continue
        run_count += 1
        db_id = q["db_id"]
        question = q["question"]
        gold_sql = q["query"]

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = get_schema_from_sqlite(get_db_path(str(data_dir), db_id))
            except FileNotFoundError as e:
                print(f"    [{idx}] SKIP — {e}")
                continue
        schema = schema_cache[db_id]
        db_path = get_db_path(str(data_dir), db_id)

        try:
            out = call_with_retry(retries, backoff,
                                  question=question, schema=schema,
                                  provider=provider, model=model,
                                  temperature=temperature, max_tokens=max_tokens)
            gen_sql = out["generated_sql"]
            total_tokens += out["usage"].get("total_tokens", 0)
        except Exception as e:  # noqa: BLE001
            print(f"    [{idx}] API ERROR (gave up after {retries} tries): {str(e)[:120]}")
            results.append({
                "spider_index": idx, "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "generated_sql": "", "api_error": str(e),
                "eval": _failed_eval(f"API error: {e}"),
            })
            continue

        gen_exec = execute_sqlite_query(gen_sql, db_path)
        gold_exec = execute_sqlite_query(gold_sql, db_path)
        evaluation = compare_results(gen_exec, gold_exec)

        status = "PASS" if evaluation["result_match"] else "FAIL"
        print(f"    [{idx + 1}/{len(dev)}] {db_id:<20} {status}  {gen_sql[:64]}")

        results.append({
            "spider_index": idx, "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "generated_sql": gen_sql,
            "usage": out["usage"],
            "gen_execution": {"success": gen_exec["success"], "row_count": gen_exec["row_count"],
                              "error": gen_exec["error"]},
            "gold_execution": {"success": gold_exec["success"], "row_count": gold_exec["row_count"],
                               "error": gold_exec["error"]},
            "eval": evaluation,
        })

        if run_count % 25 == 0:
            _save_model_run(provider, full_model, label, results, total_tokens, len(dev), output_dir)
            done = len(results)
            acc = sum(1 for r in results if r["eval"]["result_match"]) / done
            print(f"    --- checkpoint: {done}/{len(dev)} done, running result-acc {acc:.1%} ---")

    metrics = compute_metrics(results)
    out_file = _save_model_run(provider, full_model, label, results,
                               total_tokens, len(dev), output_dir, metrics)
    elapsed = time.time() - t0

    print(f"\n  {full_model}")
    print(f"    result accuracy:    {metrics.get('result_accuracy', 0):.1%}")
    print(f"    execution accuracy: {metrics.get('execution_accuracy', 0):.1%}")
    print(f"    tokens: {total_tokens:,} | time: {elapsed/60:.1f} min")
    print(f"    saved → {out_file}")

    return {
        "model": full_model,
        "label": label,
        "result_accuracy": metrics.get("result_accuracy", 0),
        "execution_accuracy": metrics.get("execution_accuracy", 0),
        "completed": len(results),
        "total_questions": len(dev),
        "total_tokens": total_tokens,
        "output_file": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DPO-teacher candidates on the Spider dev set")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model shortcuts/IDs to run (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--provider", default="wandb",
                        help="LLM provider (default: wandb / W&B Inference)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "spider_data",
                        help="Spider data dir containing dev.json + database/")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "teacher_bakeoff",
                        help="Where to save per-model result JSONs + the combined summary")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (smoke test)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0, deterministic)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max output tokens (default 1024; higher for reasoning models)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per question on API error")
    parser.add_argument("--backoff", type=float, default=5.0, help="Base backoff seconds (exponential)")
    parser.add_argument("--resume", default=None,
                        help="Resume a single interrupted model run from this result JSON "
                             "(only meaningful with one --models entry)")
    parser.add_argument("--list-models", action="store_true", help="List W&B model shortcuts and exit")
    args = parser.parse_args()

    if args.list_models:
        print("\n  W&B Inference model shortcuts:")
        for short, full in WANDB_MODELS.items():
            print(f"    {short:<18} {full}")
        print()
        sys.exit(0)

    dev = load_dev(args.data_dir)
    if args.limit:
        dev = dev[:args.limit]

    print(f"\nBenchmarking {len(args.models)} model(s) on {len(dev)} Spider dev questions")
    print(f"Provider: {args.provider} | output: {args.output_dir}")

    summaries = []
    for model in args.models:
        resume = args.resume if (args.resume and len(args.models) == 1) else None
        try:
            summaries.append(run_one_model(
                model=model, provider=args.provider, dev=dev,
                data_dir=args.data_dir, output_dir=args.output_dir,
                temperature=args.temperature, max_tokens=args.max_tokens,
                retries=args.retries, backoff=args.backoff, resume_file=resume,
            ))
        except Exception as e:  # noqa: BLE001 - one bad model shouldn't kill the whole bake-off
            print(f"\n  !! {model} failed entirely: {e}\n")
            summaries.append({"model": resolve_model(model, args.provider),
                              "label": model_label(resolve_model(model, args.provider)),
                              "result_accuracy": None, "execution_accuracy": None,
                              "error": str(e)})

    # ── Combined ranked summary ──────────────────────────────────────────────
    ranked = sorted(
        summaries,
        key=lambda s: (s.get("result_accuracy") if s.get("result_accuracy") is not None else -1),
        reverse=True,
    )
    print(f"\n{'=' * 84}")
    print(f"  TEACHER BAKE-OFF SUMMARY — {len(dev)} Spider dev questions")
    print(f"{'=' * 84}")
    print(f"  {'model':<46}{'result_acc':>12}{'exec_acc':>12}")
    print(f"  {'-' * 70}")
    for s in ranked:
        racc = f"{s['result_accuracy']:.1%}" if s.get("result_accuracy") is not None else "ERROR"
        eacc = f"{s['execution_accuracy']:.1%}" if s.get("execution_accuracy") is not None else "-"
        print(f"  {s['model']:<46}{racc:>12}{eacc:>12}")
    print(f"{'=' * 84}")

    summary_path = args.output_dir / f"spider_bakeoff_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "benchmark": "spider_dev", "questions": len(dev),
            "provider": args.provider, "ranked": ranked,
        }, f, indent=2)
    print(f"  Combined summary → {summary_path}\n")


if __name__ == "__main__":
    main()
