"""
Fine-tune Qwen2.5-Coder-7B-Instruct on Gretel's synthetic text-to-SQL dataset
using QLoRA (4-bit quantization + LoRA adapters).

Usage:
    python -m src.finetune                          # full 100K examples
    python -m src.finetune --max_samples 10000      # subset for faster iteration
    python -m src.finetune --epochs 2               # more epochs

Run this on a GPU machine (AWS g5.xlarge or similar with 24GB VRAM).
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "qwen-7b-sql-lora"

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


def format_example(example: dict) -> str:
    """
    Format a Gretel example into the Qwen chat template.
    Returns a single string that the tokenizer will process.
    """
    schema = example["sql_context"]
    question = example["sql_prompt"]
    sql = example["sql"]

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"### Database Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL Query:<|im_end|>\n"
        f"<|im_start|>assistant\n{sql}<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder-7B on text-to-SQL")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  FINE-TUNING: {BASE_MODEL}")
    print(f"  Method: QLoRA (4-bit quantization + LoRA)")
    print(f"  Max samples: {args.max_samples or 'all (100K)'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"{'=' * 70}\n")

    # --- Step 1: Load and format dataset ---
    print("Loading Gretel synthetic text-to-SQL dataset...")
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"  Using {len(dataset)} examples (subset)")
    else:
        print(f"  Using all {len(dataset)} examples")

    dataset = dataset.map(
        lambda ex: {"text": format_example(ex)},
        remove_columns=dataset.column_names,
    )

    # --- Step 2: Load tokenizer ---
    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Step 3: Load model with 4-bit quantization ---
    print(f"Loading model with 4-bit quantization...")
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
    model = prepare_model_for_kbit_training(model)

    # --- Step 4: Configure LoRA ---
    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Step 5: Training ---
    print("Starting training...\n")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    trainer.train()

    # --- Step 6: Save the LoRA adapter ---
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"\n{'=' * 70}")
    print(f"  Training complete!")
    print(f"  LoRA adapter saved to: {FINAL_MODEL_DIR}")
    print(f"  To evaluate, run: python -m src.eval_finetuned --limit 100")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
