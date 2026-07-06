"""
GRPO training for Spider Text-to-SQL — online RL with a binary SQL-execution reward,
starting from the SFT adapter. Reuses the execution reward + 4-bit model loader from
src.bird.grpo_train (both are benchmark-agnostic); uses the Spider dataset + eval prompt.

Research-informed defaults (see the holistic notes):
  - lr_scheduler = "constant"  → the #1 fix vs the failed BIRD run (cosine died by step 75,
    so the model never moved and --resume was useless).
  - beta = 0.01 (low KL)       → verifiable-reward RL benefits from a LOW KL penalty; the
    BIRD run's beta 0.05 pinned the model to SFT (KL ~0.06, no movement).
  - num_generations = 8        → better group-relative advantage estimate.
  - binary execution reward    → robust to reward hacking (correctness is objective).
  - NO vLLM                    → HF generate; vLLM pins break TRL in-env (your prior error).

    python -m src.spider.grpo_train \
        --train-json /home/phalle.y/Jenish-DPO-GRPO/train_spider.json \
        --data-dir   /home/phalle.y/Jenish-DPO-GRPO \
        --sft-adapter /scratch/phalle.y/spider_sft_adapter_7b/final_adapter \
        --output-dir /scratch/phalle.y/spider_grpo_adapter_7b --limit 2000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from src.bird.grpo_train import sql_reward_fn, load_model_for_grpo   # benchmark-agnostic
from src.spider.grpo_dataset import build_grpo_dataset

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def train(train_json, data_dir, sft_adapter, output_dir, limit, num_epochs, num_generations,
          learning_rate, lr_scheduler, beta, max_new_tokens, max_prompt_length,
          per_device_batch_size, grad_accum_steps, save_steps, resume):
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_for_grpo(BASE_MODEL, sft_adapter)
    dataset = build_grpo_dataset(train_json, data_dir, tokenizer=tokenizer, limit=limit)

    cfg = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        temperature=0.9,
        max_completion_length=max_new_tokens,
        max_prompt_length=max_prompt_length,     # Spider schemas are small; 2048 is ample
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,   # keep == num_generations (TRL divisibility)
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,           # "constant" so resume-chaining actually continues
        warmup_ratio=0.03,
        beta=beta,                                # low KL for verifiable reward
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                          reward_funcs=[sql_reward_fn], args=cfg, train_dataset=dataset)

    print("\n" + "=" * 72)
    print("  GRPO TRAINING — Spider Text-to-SQL")
    print(f"  Base:      {BASE_MODEL}")
    print(f"  SFT start: {sft_adapter or 'none (fresh LoRA)'}")
    print(f"  Examples:  {len(dataset)} | num_gen {num_generations} | lr {learning_rate} ({lr_scheduler}) | beta {beta}")
    print(f"  Output:    {output_dir}")
    print("=" * 72 + "\n")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print(f"\nTraining complete in {(time.time() - start) / 3600:.2f}h")

    final = output_dir / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"Final GRPO adapter saved to: {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GRPO training for Spider Text-to-SQL")
    ap.add_argument("--train-json", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="Spider data root (has database/)")
    ap.add_argument("--sft-adapter", type=Path, default=None, help="SFT adapter to start from (recommended)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=2000, help="Train questions to use (fits ~8h at num_gen 8)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lr-scheduler", default="constant", choices=["constant", "cosine", "linear"])
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8, help="completions/device/step — keep == --num-gen")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    train(a.train_json, a.data_dir, a.sft_adapter, a.output_dir, a.limit, a.epochs, a.num_gen,
          a.lr, a.lr_scheduler, a.beta, a.max_tokens, a.max_prompt_length, a.batch_size,
          a.grad_accum, a.save_steps, a.resume)
