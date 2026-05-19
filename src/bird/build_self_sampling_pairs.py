"""
Build DPO preference pairs from the model's OWN sampled outputs.

Uses vLLM to generate K candidates per BIRD train question, executes each
against the database, and forms (correct, wrong) preference pairs from the
same question's candidates. Output is in the same DPO training format as
src.bird.format_dpo_pairs so it can be fed directly to src.bird.dpo_train.

Addresses the "preference annotator ceiling" problem — instead of training
on pairs from Grok/DeepSeek (which 14B already matches), generate pairs
from the student model's own outputs filtered by execution correctness.

Usage:
    python -m src.bird.build_self_sampling_pairs \\
        --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \\
        --adapter     /scratch/phalle.y/bird_sft_adapter_14b/checkpoint-3100 \\
        --train-json  /scratch/phalle.y/bird_train/train/train.json \\
        --db-dir      /scratch/phalle.y/bird_train/train/train_databases/train_databases \\
        --output-file /scratch/phalle.y/results_self_sampling/bird_self_sampling_pairs.json \\
        --k 4 --temperature 0.8 \\
        --max-questions 3000

If train.json is missing, use --use-hf to load from xu3kev/BIRD-SQL-data-train.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.bird.inference import BIRDvLLMEngine, build_instruction
from src.shared.sqlite_executor import execute_sqlite_query


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_train_questions_from_json(train_json: Path, db_dir: Path) -> list[dict]:
    """Load BIRD train questions from a local train.json + sqlite databases."""
    with open(train_json) as f:
        data = json.load(f)

    out = []
    for q in data:
        db_id = q["db_id"]
        sqlite_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            continue
        # Lazy schema extraction
        conn = sqlite3.connect(str(sqlite_path))
        cur  = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        schema = "\n\n".join(row[0] for row in cur.fetchall() if row[0])
        conn.close()
        out.append({
            "question_id": q.get("question_id"),
            "db_id":       db_id,
            "question":    q["question"],
            "evidence":    q.get("evidence", ""),
            "gold_sql":    q.get("SQL") or q.get("query", ""),
            "schema":      schema,
            "db_path":     str(sqlite_path),
            "difficulty":  q.get("difficulty", "unknown"),
        })
    return out


def load_train_questions_from_hf() -> list[dict]:
    """Load BIRD train from xu3kev/BIRD-SQL-data-train. Pre-extracts schema."""
    from datasets import load_dataset
    ds = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")

    out = []
    for i, row in enumerate(ds):
        out.append({
            "question_id": i,
            "db_id":       row.get("db_id", ""),
            "question":    row["question"],
            "evidence":    row.get("evidence", ""),
            "gold_sql":    row["SQL"],
            "schema":      row["schema"],
            "db_path":     None,   # Will need user-supplied db_dir to execute
            "difficulty":  "unknown",
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pair builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(
    base_model: str,
    adapter: Path | None,
    output_file: Path,
    train_json: Path | None,
    db_dir: Path,
    k: int,
    temperature: float,
    max_questions: int | None,
    cutoff_len: int,
    use_hf: bool,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Load training questions ───────────────────────────────────────────────
    print("[1/4] Loading BIRD train questions ...")
    if use_hf:
        questions = load_train_questions_from_hf()
        # HF dataset doesn't include db_path; we need db_dir to execute
        for q in questions:
            sqlite_path = Path(db_dir) / q["db_id"] / f"{q['db_id']}.sqlite"
            q["db_path"] = str(sqlite_path) if sqlite_path.exists() else None
    else:
        questions = load_train_questions_from_json(train_json, db_dir)
    questions = [q for q in questions if q["db_path"]]
    if max_questions:
        questions = questions[:max_questions]
    print(f"      Loaded {len(questions)} questions with valid databases")

    # ── Load vLLM engine ──────────────────────────────────────────────────────
    print(f"\n[2/4] Loading vLLM engine (base={base_model}, adapter={adapter}) ...")
    engine = BIRDvLLMEngine(
        base_model=base_model,
        adapter_path=str(adapter) if adapter else None,
        max_model_len=cutoff_len,
    )

    # ── Generate K candidates per question ────────────────────────────────────
    print(f"\n[3/4] Generating K={k} samples for {len(questions)} questions ...")
    items = [(q["question"], q["schema"], q["evidence"]) for q in questions]
    start = time.time()
    all_candidates = engine.generate(items, k=k, temperature=temperature)
    print(f"      Done in {round((time.time() - start) / 60, 1)} min")

    # ── Execute, classify, build pairs ────────────────────────────────────────
    print(f"\n[4/4] Executing candidates and building pairs ...")
    pairs        = []
    skipped      = {"no_correct": 0, "all_correct": 0, "all_wrong": 0}
    per_question_stats = []

    from src.shared.evaluator import compare_results
    # 15s timeout balances: catches bad SQL with cross-joins (would hang 30s+),
    # while giving 3x headroom over typical correct query time (<5s on BIRD).
    # BIRD official eval uses 30s but that's for gold SQL only.
    EXEC_TIMEOUT = 15
    exec_start   = time.time()

    for i, (q, cands) in enumerate(zip(questions, all_candidates)):
        gold_exec = execute_sqlite_query(q["gold_sql"], q["db_path"], timeout=EXEC_TIMEOUT)
        correct, wrong = [], []
        for sql in cands:
            cand_exec = execute_sqlite_query(sql, q["db_path"], timeout=EXEC_TIMEOUT)
            ev = compare_results(cand_exec, gold_exec)
            if ev["result_match"]:
                correct.append(sql)
            else:
                wrong.append(sql)

        per_question_stats.append({
            "question_id": q["question_id"],
            "num_correct": len(correct),
            "num_wrong":   len(wrong),
        })

        if not correct:
            skipped["all_wrong"] += 1
        elif not wrong:
            skipped["all_correct"] += 1
        else:
            instruction = build_instruction(q["question"], q["schema"], q["evidence"])
            pairs.append({
                "instruction":     instruction,
                "input":           "",
                "chosen":          correct[0],
                "rejected":        wrong[0],
                "question_id":     q["question_id"],
                "db_id":           q["db_id"],
                "gold_sql":        q["gold_sql"],
                "difficulty":      q["difficulty"],
                "num_correct":     len(correct),
                "num_wrong":       len(wrong),
            })

        # Progress + intermediate save every 100 questions
        if (i + 1) % 100 == 0:
            elapsed_min = (time.time() - exec_start) / 60
            rate        = (i + 1) / max(elapsed_min, 0.01)
            eta_min     = (len(questions) - (i + 1)) / max(rate, 0.1)
            print(f"  [{i+1}/{len(questions)}] pairs={len(pairs)} "
                  f"skip_wrong={skipped['all_wrong']} skip_correct={skipped['all_correct']} "
                  f"| elapsed={elapsed_min:.1f}min ETA={eta_min:.1f}min")
            # Intermediate save so a SLURM timeout doesn't destroy all pairs
            with open(output_file, "w") as f:
                json.dump(
                    [{"instruction": p["instruction"], "input": "", "chosen": p["chosen"], "rejected": p["rejected"]}
                     for p in pairs],
                    f, indent=2,
                )

    # ── Save ──────────────────────────────────────────────────────────────────
    # Keep only the DPO-relevant keys in the trainer-compatible file
    dpo_examples = [
        {"instruction": p["instruction"], "input": "", "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ]
    with open(output_file, "w") as f:
        json.dump(dpo_examples, f, indent=2)

    # Metadata file alongside (for analysis / reproducibility)
    meta_file = output_file.parent / (output_file.stem + "_metadata.json")
    with open(meta_file, "w") as f:
        json.dump({
            "base_model": base_model,
            "adapter":    str(adapter) if adapter else None,
            "k":          k,
            "temperature": temperature,
            "num_questions_attempted": len(questions),
            "num_pairs_built":          len(pairs),
            "skipped":                  skipped,
            "per_question_stats":       per_question_stats,
            "pairs_with_metadata":      pairs,   # full pair entries with provenance
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  SELF-SAMPLING DONE")
    print(f"{'=' * 70}")
    print(f"  Questions attempted:   {len(questions)}")
    print(f"  Pairs built:           {len(pairs)}")
    print(f"  Skipped all wrong:     {skipped['all_wrong']}")
    print(f"  Skipped all correct:   {skipped['all_correct']}")
    print(f"  Yield rate:            {len(pairs) / max(1, len(questions)) * 100:.1f}%")
    print(f"\n  Saved pairs:    {output_file}")
    print(f"  Saved metadata: {meta_file}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Build self-sampling DPO pairs with vLLM")
    parser.add_argument("--base-model",    type=str,  required=True)
    parser.add_argument("--adapter",       type=Path, default=None,
                        help="LoRA adapter path (e.g., the SFT adapter)")
    parser.add_argument("--output-file",   type=Path, required=True,
                        help="Where to save the DPO-format pairs JSON")
    parser.add_argument("--train-json",    type=Path, default=None,
                        help="Path to train.json (omit + use --use-hf for HF dataset)")
    parser.add_argument("--db-dir",        type=Path, required=True,
                        help="BIRD train_databases directory")
    parser.add_argument("--k",             type=int,   default=4)
    parser.add_argument("--temperature",   type=float, default=0.8)
    parser.add_argument("--max-questions", type=int,   default=None,
                        help="Cap number of training questions (smoke test / budget)")
    parser.add_argument("--cutoff-len",    type=int,   default=8192)
    parser.add_argument("--use-hf",        action="store_true",
                        help="Load questions from xu3kev/BIRD-SQL-data-train HF dataset")
    args = parser.parse_args()

    if not args.use_hf and not args.train_json:
        parser.error("Must provide --train-json or --use-hf")

    build_pairs(
        base_model=args.base_model,
        adapter=args.adapter,
        output_file=args.output_file,
        train_json=args.train_json,
        db_dir=args.db_dir,
        k=args.k,
        temperature=args.temperature,
        max_questions=args.max_questions,
        cutoff_len=args.cutoff_len,
        use_hf=args.use_hf,
    )


if __name__ == "__main__":
    main()
