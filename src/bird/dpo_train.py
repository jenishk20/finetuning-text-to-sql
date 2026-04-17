"""
DPO training for BIRD Text-to-SQL using frontier preference pairs.

Loads the SFT checkpoint (bird_sft_adapter_1) and applies DPO on top
using clear_preference pairs built from Grok vs DeepSeek frontier runs.

Why DPO after SFT:
  - SFT teaches the task (generate SQL from schema + question)
  - DPO refines quality by learning which SQL patterns are preferred
  - Frontier preference pairs provide high-quality signal (one correct, one wrong)

Run:
    python -m src.bird.dpo_train \
        --pairs-file /scratch/phalle.y/results_frontier_pairs/bird_frontier_dpo_data.json \
        --sft-adapter /home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1 \
        --output-dir /scratch/phalle.y/bird_frontier_dpo_adapter

    # Resume from checkpoint
    python -m src.bird.dpo_train ... --resume /scratch/phalle.y/bird_frontier_dpo_adapter/checkpoint-100

Requirements:
    pip install trl>=0.12 transformers peft bitsandbytes accelerate datasets
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dpo_dataset(pairs_file: Path, tokenizer) -> Dataset:
    """
    Load bird_frontier_dpo_data.json and convert to HuggingFace Dataset.

    DPOTrainer expects columns: prompt, chosen, rejected.
    We apply the chat template to the prompt so it matches what the
    tokenizer saw during SFT training.
    """
    with open(pairs_file) as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} preference pairs from {pairs_file.name}")

    prompts, chosens, rejecteds = [], [], []

    for pair in pairs:
        instruction = pair["instruction"]

        # Apply chat template to match SFT training format
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(prompt)
        chosens.append(pair["chosen"])
        rejecteds.append(pair["rejected"])

    dataset = Dataset.from_dict({
        "prompt":   prompts,
        "chosen":   chosens,
        "rejected": rejecteds,
    })

    print(f"Dataset ready: {len(dataset)} examples")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def load_model_for_dpo(base_model: str, sft_adapter: Path | None):
    """
    Load Qwen 7B in 4-bit NF4. Loads SFT adapter as the starting point for DPO.
    DPOTrainer manages the reference model internally.
    """
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left-padding

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

    if sft_adapter and Path(sft_adapter).exists():
        print(f"Loading SFT adapter from {sft_adapter}...")
        model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)
        print("SFT adapter loaded — DPO will train on top of SFT weights")
    else:
        print("No SFT adapter found — adding fresh LoRA (not recommended for DPO)")
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


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    pairs_file: Path,
    sft_adapter: Path | None,
    output_dir: Path,
    num_epochs: int = 1,
    beta: float = 0.05,
    learning_rate: float = 5e-5,
    cutoff_len: int = 4096,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 8,
    save_steps: int = 50,
    resume_from_checkpoint: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer ────────────────────────────────────────────────
    model, tokenizer = load_model_for_dpo(BASE_MODEL, sft_adapter)

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\nBuilding DPO dataset...")
    dataset = build_dpo_dataset(pairs_file, tokenizer)

    # ── DPO config ────────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=str(output_dir),

        # DPO-specific
        beta=beta,                              # KL penalty — stay close to SFT
        max_length=cutoff_len,                  # max total sequence length
        max_prompt_length=cutoff_len // 2,      # max prompt tokens

        # Training
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

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
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 70)
    print("  DPO TRAINING — BIRD Text-to-SQL (Frontier Pairs)")
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  SFT adapter:   {sft_adapter or 'none (not recommended)'}")
    print(f"  Pairs file:    {pairs_file.name}")
    print(f"  Dataset size:  {len(dataset)}")
    print(f"  Beta (KL):     {beta}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Cutoff len:    {cutoff_len}")
    print(f"  Output dir:    {output_dir}")
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
    parser = argparse.ArgumentParser(description="DPO training for BIRD Text-to-SQL")

    parser.add_argument("--pairs-file",  type=Path, required=True,
                        help="Path to bird_frontier_dpo_data.json (output of format_dpo_pairs.py)")
    parser.add_argument("--sft-adapter", type=Path, default=None,
                        help="Path to SFT LoRA adapter (bird_sft_adapter_1)")
    parser.add_argument("--output-dir",  type=Path,
                        default=Path("/scratch/phalle.y/bird_frontier_dpo_adapter"),
                        help="Where to save DPO adapter checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--beta",        type=float, default=0.05,
                        help="DPO beta — KL penalty, lower = stay closer to SFT (default: 0.05)")
    parser.add_argument("--lr",          type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--cutoff-len",  type=int,   default=8192,
                        help="Max sequence length — use 8192 on H200, 4096 on A100, 1536 on A10G (default: 8192)")
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--save-steps",  type=int,   default=50,
                        help="Save checkpoint every N steps (default: 50 for 2h session limit)")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint dir to resume from")

    args = parser.parse_args()

    train(
        pairs_file=args.pairs_file,
        sft_adapter=args.sft_adapter,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        beta=args.beta,
        learning_rate=args.lr,
        cutoff_len=args.cutoff_len,
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
    )
