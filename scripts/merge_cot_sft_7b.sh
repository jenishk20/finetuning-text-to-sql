#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --job-name=merge-cot-sft
#SBATCH --mem=60GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/phalle.y/merge_cot_sft_%j.out
#SBATCH --error=/scratch/phalle.y/merge_cot_sft_%j.err

# py3.10 venv with a RECENT peft — understands the adapter's newer config fields
# (alora_invocation_tokens etc.), so no config-stripping needed. Self-contained venv.
PYBIN=/scratch/phalle.y/mergeenv310/bin/python

export HF_HOME=/scratch/phalle.y/hf_cache
export TRANSFORMERS_CACHE=/scratch/phalle.y/hf_cache

"$PYBIN" - <<'PYEOF'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE    = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER = "/scratch/phalle.y/bird_cot_sft_adapter_7b/final_adapter"
OUT     = "/scratch/phalle.y/bird_cot_sft_merged_7b"

print("Loading base in bf16 (GPU) ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
print("Merging LoRA ...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
print(f"Saving merged fp16 model -> {OUT}", flush=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("DONE. Merged model ready for upload + GGUF.", flush=True)
PYEOF
