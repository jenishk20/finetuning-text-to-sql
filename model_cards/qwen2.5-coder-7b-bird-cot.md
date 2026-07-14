---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
pipeline_tag: text-generation
library_name: transformers
language:
- en
tags:
- text-to-sql
- sql
- bird
- chain-of-thought
- reasoning
- qwen2.5-coder
- gguf
- llama.cpp
datasets:
- jk200201/bird-cot-sft
model-index:
- name: qwen2.5-coder-7b-bird-cot
  results:
  - task:
      type: text-generation
      name: Text-to-SQL
    dataset:
      type: bird
      name: BIRD (dev)
    metrics:
    - type: accuracy
      name: Result accuracy (greedy)
      value: 52.1
    - type: accuracy
      name: Result accuracy (self-consistency, K=8)
      value: 58.5
---

# Qwen2.5-Coder-7B — BIRD CoT (Text-to-SQL)

A 7B text-to-SQL model that **reasons step-by-step over a database schema, then writes the SQL**. Fine-tuned from `Qwen/Qwen2.5-Coder-7B-Instruct` by distilling *execution-verified* chain-of-thought solutions.

On **BIRD dev** (messy, real-world schemas — the hard text-to-SQL benchmark) it reaches **52.1%** result accuracy greedy, and **58.5%** with self-consistency (Best-of-N, K=8) — **surpassing single-shot frontier models at a fraction of the size.**

## Results — BIRD dev (execution result accuracy)

| Model | Result accuracy |
|---|---|
| Qwen2.5-Coder-7B-Instruct (base) | ~27% |
| **This model — greedy (1 sample)** | **52.1%** |
| **This model — self-consistency (K=8)** | **58.5%** |
| Grok-4 (single-shot) | 55.4% |
| DeepSeek-V3 (single-shot) | 54.7% |

*Result accuracy* = the generated SQL executes to the **same rows** as the gold query (BIRD's official execution metric). Frontier numbers are single-shot; the 58.5% uses K=8 self-consistency (8× inference).

## Usage

Prompt the model to reason step-by-step; it returns the reasoning followed by a fenced SQL block. **Take the last ```sql``` block** as the query.

```python
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "jk200201/qwen2.5-coder-7b-bird-cot"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

SYSTEM = ("You are an expert SQLite query writer. Reason step by step about the schema "
          "and the question, then output the final query in a ```sql code block.")

def build_prompt(schema, question):
    return ("Given the database schema and question, work out the correct SQLite query step by step.\n\n"
            f"Database Schema:\n{schema}\n\nQuestion: {question}\n\n"
            "Think step by step:\n1. Which tables and columns are relevant?\n"
            "2. What joins, filters, grouping, and ordering are needed?\n3. Handle edge cases.\n\n"
            "Then give the final answer as:\n```sql\n<final query>\n```")

schema   = "CREATE TABLE singer (Singer_ID INT, Name TEXT, Age INT);"
question = "How many singers are older than 40?"
messages = [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": build_prompt(schema, question)}]

text   = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=512, do_sample=False)
resp   = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

sql = re.findall(r"```(?:sql)?\s*(.*?)```", resp, re.DOTALL)[-1].strip()
print(sql)
```

**For best accuracy (58.5%):** sample K=8 at `temperature=0.8`, execute each candidate, and take the majority result (self-consistency).

## Run locally (GGUF / Ollama / llama.cpp)

Quantized GGUF builds (`Q4_K_M`, `Q5_K_M`, `Q8_0`) are provided for CPU/laptop use:

```bash
ollama run jk200201/qwen2.5-coder-7b-bird-cot
# or with llama.cpp:
./llama-cli -m qwen2.5-coder-7b-bird-cot.Q4_K_M.gguf -sys "$SYSTEM" -p "$PROMPT"
```

## Training

- **Method — reasoning distillation (CoT-SFT):** a strong teacher (Qwen3-Coder-480B) generated step-by-step CoT solutions on BIRD train; only **execution-verified-correct** chains were kept (**5,593** examples), then supervised-fine-tuned into the 7B. Distilling the teacher's *reasoning* generalized across BIRD's cross-domain dev databases better than distilling SQL answers directly.
- **Config:** QLoRA (4-bit NF4, LoRA r=32, α=64), 2 epochs, LR 2e-4 cosine, max seq len 8192.
- **Data:** [`jk200201/bird-cot-sft`](https://huggingface.co/datasets/jk200201/bird-cot-sft)

## Limitations

- Tuned for **BIRD-style** analytic SQL over realistic schemas; unusual dialects/domains may need adaptation. Emits **SQLite** dialect.
- Greedy (52.1%) is the deployable single-shot; the 58.5% figure requires K=8 self-consistency (8× inference cost).
- A 7B model — always review generated SQL before running it on production data.

## Citation

```bibtex
@misc{qwen25coder7b_bird_cot_2026,
  title  = {Qwen2.5-Coder-7B BIRD CoT: reasoning distillation for text-to-SQL},
  author = {Jenish Kothari},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-cot}}
}
```

## Acknowledgements
Base: Qwen2.5-Coder (Alibaba Qwen). Teacher: Qwen3-Coder-480B (W&B Inference). Benchmark: [BIRD](https://bird-bench.github.io/).
