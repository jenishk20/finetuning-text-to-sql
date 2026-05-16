"""
SFT training for BIRD Text-to-SQL.

Trains a fresh LoRA adapter on top of Qwen2.5-Coder (7B or 14B) using
gold SQL from BIRD train. This is the SFT stage that precedes DPO.

Run:
    python -m src.bird.sft_train \\
        --sft-data    /scratch/phalle.y/bird_sft_data.json \\
        --output-dir  /scratch/phalle.y/bird_sft_adapter_14b \\
        --base-model  Qwen/Qwen2.5-Coder-14B-Instruct \\
        --epochs 3 --cutoff-len 8192

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def build_dataset(sft_file: Path, tokenizer) -> Dataset:
    """Load bird_sft_data.json and convert into single-text format for SFTTrainer."""
    with open(sft_file) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} SFT examples from {sft_file.name}")

    texts = []
    for ex in examples:
        messages = [
            {"role": "user",      "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def load_model(base_model: str):
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        attn_implementation="flash_attention_2" if _flash_attn_available() else "eager",
    )
    model = prepare_model_for_kbit_training(model)

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


def train(
    sft_data: Path,
    output_dir: Path,
    base_model: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    cutoff_len: int = 8192,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 8,
    save_steps: int = 100,
    resume_from_checkpoint: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(base_model)
    dataset          = build_dataset(sft_data, tokenizer)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=cutoff_len,
        packing=False,

        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,

        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 70)
    print("  SFT TRAINING — BIRD Text-to-SQL")
    print(f"  Base model:    {base_model}")
    print(f"  SFT data:      {sft_data.name}")
    print(f"  Dataset size:  {len(dataset)}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Cutoff len:    {cutoff_len}")
    print(f"  Batch/grad:    {per_device_batch_size} × {grad_accum_steps}")
    print(f"  Output dir:    {output_dir}")
    print("=" * 70 + "\n")

    start   = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = round((time.time() - start) / 3600, 2)
    print(f"\nTraining complete in {elapsed}h")

    final_path = output_dir / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final SFT adapter saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training for BIRD Text-to-SQL")
    parser.add_argument("--sft-data",   type=Path, required=True,
                        help="Path to bird_sft_data.json (output of build_sft_data.py)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save SFT adapter checkpoints")
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-Coder-14B-Instruct",
                        help="HF base model ID")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--cutoff-len", type=int,   default=8192,
                        help="Max sequence length (8192 on H200, 4096 on A100)")
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--save-steps", type=int,   default=100)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        sft_data=args.sft_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        cutoff_len=args.cutoff_len,
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
    )
