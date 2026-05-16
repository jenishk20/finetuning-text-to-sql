---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
library_name: peft
tags:
- text-to-sql
- sql
- code-generation
- spider
- dpo
- qwen
- qwen2.5
- text-generation
language:
- en
pipeline_tag: text-generation
datasets:
- jk200201/spider-dpo-1040
---

# Qwen2.5-Coder-7B Spider-DPO

A LoRA adapter for **Qwen2.5-Coder-7B-Instruct** fine-tuned with DPO that achieves **78.2% on Spider V1 dev** — outperforming Grok-4 (73.7%) and DeepSeek V3 (71.8%) despite being a 7B model.

## Results

| Model | Spider V1 dev | Parameters |
|---|---|---|
| **Qwen2.5-Coder-7B + Spider-DPO (this model)** | **78.2%** | 7B |
| Grok-4 (frontier baseline) | 73.7% | unknown (very large) |
| DeepSeek-V3 (frontier baseline) | 71.8% | 671B (37B active MoE) |
| Qwen2.5-Coder-7B base | ~50% | 7B |

### Cross-benchmark transfer

| Benchmark | Score |
|---|---|
| Spider V1 dev (in-domain) | **78.2%** |

For real-world database queries (BIRD-style schemas with evidence), use the companion model: [`jk200201/qwen2.5-coder-7b-bird-dpo`](https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-dpo).

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER    = "jk200201/qwen2.5-coder-7b-sql-dpo"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

schema   = "CREATE TABLE users (id INT, name TEXT, country TEXT);"
question = "How many users are from Japan?"

prompt = f"""Convert the following natural language question into a valid SQL query.

Database Schema:
{schema}

Question: {question}

Return only the SQL query with no explanation."""

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    return_tensors="pt", add_generation_prompt=True
).to(model.device)

out = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
sql = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
print(sql)
```

## Training Details

**The novel idea**: rather than human-annotated preferences, this model uses **automatically generated preference pairs from frontier model disagreements** — total cost: ~$25 of OpenRouter API calls.

### Pipeline

1. Run **Grok-4** and **DeepSeek-V3** on Spider dev set (1,034 questions).
2. Compare against gold SQL question-by-question. Where one frontier model is right and the other wrong → preference pair (the correct SQL is "chosen", the wrong one "rejected").
3. SFT Qwen2.5-Coder-7B on Spider train gold SQL (QLoRA r=32, α=64, NF4 4-bit, 3 epochs).
4. DPO on **1,040 clear-preference pairs** on top of SFT (β=0.1, 2 epochs).

### Hyperparameters

| Stage | Setting |
|---|---|
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| SFT epochs | 3, LR 2e-4 cosine |
| DPO epochs | 2, LR 5e-5 cosine, β=0.1 |

### Training data

[`jk200201/spider-dpo-1040`](https://huggingface.co/datasets/jk200201/spider-dpo-1040) — 1,040 preference pairs built from Grok-4 vs DeepSeek-V3 disagreements on Spider dev.

### Hardware

AWS EC2 g5.xlarge (NVIDIA A10G 24GB VRAM). Training time: ~3h total.

## Limitations

- Designed for **Spider-style** queries: academic-style English, clean schemas, single SQLite dialect
- For real-world messy databases with domain knowledge ("BIRD-style"), use [`jk200201/qwen2.5-coder-7b-bird-dpo`](https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-dpo)
- 4-bit quantized — for highest accuracy use bf16 base model
- Trained only on English questions

## Citation

```bibtex
@misc{kothari2026qwenspiderdpo,
  author       = {Kothari, Jenish},
  title        = {Qwen2.5-Coder-7B Spider-DPO: A 7B Model that Beats Frontier Models on Spider via Frontier-Disagreement DPO},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo}},
}
```
