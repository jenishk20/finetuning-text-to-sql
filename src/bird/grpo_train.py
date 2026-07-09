"""
GRPO training for BIRD Text-to-SQL.

Uses TRL's GRPOTrainer with a SQL execution reward function.
Starts from the SFT checkpoint (bird_sft_adapter_1) and applies
online RL — the model generates SQL, executes it, and learns
from which candidates were correct.

Why GRPO instead of DPO:
  - Binary SQL reward (correct/incorrect) is perfect for GRPO
  - No pre-built preference pairs needed — reward is live execution
  - Model improves continuously — harder questions become solvable
  - No distribution shift between generation and training

Run:
    python -m src.bird.grpo_train \
        --train-json /scratch/phalle.y/bird_train/train/train.json \
        --db-dir /scratch/phalle.y/bird_train/train/train_databases/train_databases \
        --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
        --output-dir /scratch/phalle.y/bird_grpo_adapter

Requirements:
    pip install trl>=0.12 transformers peft bitsandbytes accelerate datasets
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import threading
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.bird.grpo_dataset import build_grpo_dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SQL_TIMEOUT = 30  # seconds — match BIRD eval standard


# ─────────────────────────────────────────────────────────────────────────────
# SQL EXECUTION REWARD
# ─────────────────────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    """Strip markdown fences and think blocks from model output."""
    # Remove <think>...</think> blocks (DeepSeek-style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract from ```sql ... ``` or ``` ... ```
    fence = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if not text.endswith(";"):
        text += ";"
    return text


def execute_sql(sql: str, db_path: str, timeout: int = SQL_TIMEOUT) -> dict:
    """Execute SQL with hard timeout via conn.interrupt()."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()
        try:
            cursor = conn.cursor()
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            columns, rows = [], []
            for stmt in statements:
                cursor.execute(stmt)
                if cursor.description:
                    columns = [d[0] for d in cursor.description]
                    rows = cursor.fetchall()
            return {"success": True, "columns": columns, "rows": rows}
        finally:
            timer.cancel()
    except Exception as e:
        return {"success": False, "columns": [], "rows": [], "error": str(e)}
    finally:
        if conn:
            conn.close()


def normalize_rows(rows: list) -> set:
    return {tuple(str(v) for v in row) for row in rows}


def results_match(gen_result: dict, gold_result: dict) -> bool:
    if not gen_result["success"] or not gold_result["success"]:
        return False
    return normalize_rows(gen_result["rows"]) == normalize_rows(gold_result["rows"])


def sql_reward_fn(completions, gold_sql, db_path, **kwargs) -> list[float]:
    """
    Reward function for GRPOTrainer.

    Called with a batch of completions for the same prompt.
    Returns a list of floats: 1.0 if SQL is correct, 0.0 otherwise.

    GRPOTrainer passes extra dataset columns as kwargs — gold_sql and
    db_path come from the dataset rows.
    """
    rewards = []

    # gold_sql and db_path may be lists (one per example in the group)
    # or scalars — normalize to lists
    if isinstance(gold_sql, str):
        gold_sql = [gold_sql] * len(completions)
    if isinstance(db_path, str):
        db_path = [db_path] * len(completions)

    # Cache gold results per (db, gold_sql): the N candidates for one prompt all
    # share the same gold query, so executing it once instead of N times roughly
    # halves the SQL work that dominates step time.
    gold_cache: dict = {}

    for completion, g_sql, d_path in zip(completions, gold_sql, db_path):
        try:
            # completion is the raw model output text
            gen_sql = extract_sql(completion)
            gen_result  = execute_sql(gen_sql, d_path)
            cache_key = (d_path, g_sql)
            if cache_key not in gold_cache:
                gold_cache[cache_key] = execute_sql(g_sql, d_path)
            gold_result = gold_cache[cache_key]
            reward = 1.0 if results_match(gen_result, gold_result) else 0.0
        except Exception:
            reward = 0.0
        rewards.append(reward)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_grpo(base_model: str, sft_adapter: Path | None):
    """
    Load Qwen 7B in 4-bit NF4. If sft_adapter is provided, load the SFT
    LoRA weights on top as the starting point for GRPO.
    """
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit NF4...")
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
        attn_implementation="flash_attention_2" if _flash_attn_available() else "eager",
    )
    model = prepare_model_for_kbit_training(model)

    if sft_adapter is not None:
        # Fail loud: a path was requested but isn't a usable adapter. Two common
        # causes: (1) on HPC, /home is often NOT mounted on GPU compute nodes;
        # (2) the path points at the parent dir, not the actual adapter subdir
        # (e.g. final_adapter/ or checkpoint-*/). Silently falling back to a fresh
        # LoRA wastes the entire job and isn't the intended experiment.
        if not (Path(sft_adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No adapter_config.json found under:\n    {sft_adapter}\n"
                f"--sft-adapter must point at the PEFT adapter dir itself — often a "
                f"'final_adapter/' or 'checkpoint-*/' subdir. Also ensure it's on "
                f"/scratch (/home is not mounted on GPU compute nodes)."
            )
        print(f"Loading SFT adapter from {sft_adapter}...")
        model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)
        print("SFT adapter loaded — GRPO will train on top of SFT weights")
    else:
        print("No --sft-adapter provided — adding fresh LoRA for GRPO training")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


def _flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_json: Path,
    db_dir: Path,
    sft_adapter: Path | None,
    output_dir: Path,
    num_epochs: int = 1,
    num_generations: int = 8,
    limit: int | None = None,
    learning_rate: float = 1e-6,
    lr_scheduler: str = "cosine",
    beta: float = 0.05,
    max_new_tokens: int = 512,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 8,
    save_steps: int = 50,
    resume_from_checkpoint: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer ────────────────────────────────────────────────
    model, tokenizer = load_model_for_grpo(BASE_MODEL, sft_adapter)

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\nBuilding GRPO dataset...")
    dataset = build_grpo_dataset(train_json, db_dir, tokenizer=tokenizer)
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
        print(f"Limited to {len(dataset)} examples (--limit {limit}) to fit the SLURM window")
    print(f"Dataset ready: {len(dataset)} examples")

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),

        # Core GRPO
        num_generations=num_generations,       # N candidates per question
        temperature=0.9,                        # diversity in generated candidates
        max_completion_length=max_new_tokens,   # generated SQL length cap
        max_prompt_length=3072,                 # keep long BIRD schemas (default 512 truncates)

        # Training
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,   # "constant" recommended — cosine over few steps decays to ~0 and freezes the model
        warmup_ratio=0.05,

        # KL penalty — lower (0.01-0.03) gives more room to move; verifiable-reward RL likes low KL
        beta=beta,

        # Memory
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,

        # Logging + saving
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",

        # Reproducibility
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[sql_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print("\n" + "=" * 70)
    print("  GRPO TRAINING — BIRD Text-to-SQL")
    print(f"  Base model:      {BASE_MODEL}")
    print(f"  SFT adapter:     {sft_adapter or 'none (fresh LoRA)'}")
    print(f"  Dataset size:    {len(dataset)}")
    print(f"  Num generations: {num_generations}")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Output dir:      {output_dir}")
    print("=" * 70 + "\n")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = round((time.time() - start) / 3600, 2)
    print(f"\nTraining complete in {elapsed}h")

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = output_dir / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final adapter saved to: {final_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for BIRD Text-to-SQL")

    parser.add_argument("--train-json",   type=Path, required=True,
                        help="Path to BIRD train.json")
    parser.add_argument("--db-dir",       type=Path, required=True,
                        help="Path to BIRD train_databases directory")
    parser.add_argument("--sft-adapter",  type=Path, default=None,
                        help="Path to SFT LoRA adapter checkpoint (recommended)")
    parser.add_argument("--output-dir",   type=Path, default=Path("/scratch/phalle.y/bird_grpo_adapter"),
                        help="Where to save GRPO adapter checkpoints")
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--num-gen",      type=int,   default=8,
                        help="Number of SQL candidates per question (GRPO group size)")
    parser.add_argument("--limit",        type=int,   default=None,
                        help="Cap the number of train questions (to fit the SLURM window)")
    parser.add_argument("--lr",           type=float, default=1e-6)
    parser.add_argument("--lr-scheduler", type=str,   default="cosine",
                        choices=["cosine", "constant", "linear"],
                        help="'constant' recommended — cosine over few steps decays to ~0 and freezes the model")
    parser.add_argument("--beta",         type=float, default=0.05,
                        help="KL penalty; lower (0.01-0.03) gives more room to move on verifiable rewards")
    parser.add_argument("--max-tokens",   type=int,   default=512)
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--save-steps",   type=int,   default=50,
                        help="Checkpoint every N optimizer steps (frequent = evaluable if killed)")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint dir to resume from (e.g. output_dir/checkpoint-200)")

    args = parser.parse_args()

    train(
        train_json=args.train_json,
        db_dir=args.db_dir,
        sft_adapter=args.sft_adapter,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        num_generations=args.num_gen,
        limit=args.limit,
        learning_rate=args.lr,
        lr_scheduler=args.lr_scheduler,
        beta=args.beta,
        max_new_tokens=args.max_tokens,
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
    )
