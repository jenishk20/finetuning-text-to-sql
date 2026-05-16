---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
library_name: peft
tags:
- text-to-sql
- sql
- code-generation
- bird
- bird-sql
- spider
- dpo
- qwen
- qwen2.5
- text-generation
language:
- en
pipeline_tag: text-generation
datasets:
- birdsql/bird_mini_dev
---

# Qwen2.5-Coder-7B BIRD-DPO

A LoRA adapter for **Qwen2.5-Coder-7B-Instruct** fine-tuned for real-world Text-to-SQL with **DPO using frontier model disagreements**. Achieves **50.3% on the BIRD dev set** — a +23.3pp improvement over the base model on real-world database queries with messy schemas and domain-specific evidence.

## Results

| Benchmark | Metric | Score |
|---|---|---|
| **BIRD dev** (1,534 Q) | Result accuracy | **50.3%** |
| **BIRD mini-dev** (500 Q) | Result accuracy | **44.4%** |
| BIRD mini-dev — Simple | Result accuracy | 62.8% |
| BIRD mini-dev — Moderate | Result accuracy | 38.8% |
| BIRD mini-dev — Challenging | Result accuracy | 31.4% |
| **Spider V1 dev** (1,034 Q, cross-eval) | Result accuracy | **75.9%** |

**Cross-domain transfer**: trained purely on BIRD, this adapter scores **75.9% on Spider** — only 2.3pp below a Spider-specific adapter (78.2%). Strong evidence that DPO from frontier disagreements generalizes across SQL benchmarks.

| Model | BIRD dev |
|---|---|
| Qwen2.5-Coder-7B base | 27.0% |
| Qwen2.5-Coder-7B + SFT only | ~35% |
| **Qwen2.5-Coder-7B + SFT + BIRD-DPO (this model)** | **50.3%** |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER    = "jk200201/qwen2.5-coder-7b-bird-dpo"

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
evidence = ""  # optional domain hint for BIRD-style queries

prompt = f"""Convert the following natural language question into a valid SQL query.

Database Schema:
{schema}

{f'External Knowledge:{chr(10)}{evidence}{chr(10)}{chr(10)}' if evidence.strip() else ''}Question: {question}

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

1. Run **Grok-4.1-fast** and **DeepSeek-V3.2** on BIRD train (9,428 questions). Both score ~53%.
2. Compare results question-by-question. Where one model is right and the other wrong → preference pair (1,219 pairs from BIRD).
3. SFT Qwen2.5-Coder-7B on BIRD train gold SQL (QLoRA r=32, α=64, NF4 4-bit, 3 epochs).
4. DPO on the 1,219 clear-preference pairs on top of SFT (β=0.05, 1 epoch, cutoff=8192).

### Hyperparameters

| Stage | Setting |
|---|---|
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| SFT epochs | 3, LR 2e-4 cosine |
| DPO epochs | 1, LR 5e-5 cosine, β=0.05 |
| Cutoff length | 8,192 tokens (H200 80GB) |

### Hardware

Northeastern Discovery HPC — single NVIDIA H200 80GB. Training time: ~1h 15min.

## Key Finding — What Doesn't Work

Initially trained with 4,677 pairs (1,219 clear-preference + 3,458 judge-resolved style pairs). This **regressed to 40.7%** (-9.6pp).

**Lesson**: judge-resolved pairs where both models are correct but write different SQL carry zero correctness signal for BIRD's result-accuracy metric. They dilute training and hurt performance. Only use pairs where one model is right and one is wrong.

## Limitations

- BIRD test set was not used (hidden); evaluation is on the public dev set
- Real-world databases with very long schemas (>8K tokens) get truncated
- Optimized for SQLite syntax (BIRD format); MySQL/PostgreSQL outputs may need adaptation
- Trained only on English questions

## Related Models

- **Spider version** of this approach: [`jk200201/qwen2.5-coder-7b-sql-dpo`](https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo) — 78.2% on Spider V1, also beats Grok-4 and DeepSeek V3

## Citation

If you use this model, please cite:

```bibtex
@misc{kothari2026qwenbirddpo,
  author       = {Kothari, Jenish},
  title        = {Qwen2.5-Coder-7B BIRD-DPO: Frontier-Disagreement DPO for Real-World Text-to-SQL},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-dpo}},
}
```
