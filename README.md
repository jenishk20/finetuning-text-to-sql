# 🦆 LocalSQL — ask your database questions in plain English, on your own machine

A fine-tuned **Qwen2.5-Coder-7B** that turns natural-language questions into SQL and runs them against your database — **fully local, no API keys, your data never leaves your machine.** Small enough for a single consumer GPU, yet it holds its own against frontier models 100× its size.

```bash
python -m src.text2sql --db chinook.sqlite --q "Which 5 artists have the most albums?"
```
```
── SQL ────────────────────────────────────────────
SELECT ar.Name, COUNT(al.AlbumId) AS album_count
FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId
GROUP BY ar.ArtistId ORDER BY album_count DESC LIMIT 5;

── Results (5 rows) ──────────────────────────────
| Name         |   album_count |
|--------------|---------------|
| Iron Maiden  |            21 |
| Led Zeppelin |            14 |
| ...          |           ... |
```

Point it at **any SQLite file** — it reads the schema automatically, generates the query, runs it, and shows you the rows.

---

## Why this exists

Most "text-to-SQL" tools wrap a cloud LLM (OpenAI, etc.): API keys, per-query cost, and your schema + data leaving your machine. This is the opposite — an **open 7B you run yourself**, trained for **~$25**, competitive with much larger models.

### Benchmarks

| Model | Spider V1 dev | Size |
|-------|:---:|:---:|
| **Qwen2.5-Coder-7B + this adapter** | **78.2%** | **7B · runs local** |
| Grok-4 *(version tested)* | 73.7% | very large · API-only |
| DeepSeek-V3 *(version tested)* | 71.8% | 671B MoE · API-only |
| Qwen2.5-Coder-7B (base) | ~50% | 7B |

> Execution accuracy (result-set match) on the 1,034-question Spider V1 dev set. Frontier baselines were measured **in early 2026** against the Grok-4 / DeepSeek-V3 versions available then — frontier models keep improving, so read this as *"a 7B competing in the frontier's weight class,"* not a permanent leaderboard. The full eval harness ships in this repo (`src/spider/`) so you can re-run any baseline yourself.

For comparison, this matches **DIN-SQL + GPT-3 Codex (78.2%)** on the Spider V1 dev leaderboard — from a 7B you run on your own GPU.

Spider is clean, academic SQL. On messier real-world databases with domain hints ("BIRD-style") the 7B is *competitive but not ahead* of frontier — we don't hide that; use the companion adapter below.

---

## Install

```bash
git clone https://github.com/jenishk20/finetuning-text-to-sql
cd finetuning-text-to-sql
pip install -r requirements-gpu.txt        # torch, transformers, peft, bitsandbytes, tabulate
```
Needs a CUDA GPU (~6 GB VRAM in 4-bit). First run downloads the base model + adapter from Hugging Face.

## Usage

```bash
# ask a question (4-bit, auto-runs the SQL)
python -m src.text2sql --db mydata.sqlite --q "total revenue per month in 2023"

# just generate SQL, don't execute it
python -m src.text2sql --db mydata.sqlite --q "..." --no-exec

# higher accuracy in bf16 (more VRAM)
python -m src.text2sql --db mydata.sqlite --q "..." --bf16

# BIRD-style adapter + a domain hint
python -m src.text2sql --db mydata.sqlite --q "..." \
    --adapter jk200201/qwen2.5-coder-7b-bird-dpo --evidence "revenue = price * quantity"
```

As a library:
```python
from src.text2sql import load_model, predict
model, tok = load_model()
r = predict("mydata.sqlite", "top 5 customers by revenue", model, tok)
print(r["sql"]); print(r["rows"])
```

## Models

| Adapter | Best for | Link |
|---|---|---|
| `jk200201/qwen2.5-coder-7b-sql-dpo` | General / Spider-style (default) | [HF](https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo) |
| `jk200201/qwen2.5-coder-7b-bird-dpo` | Messy real-world schemas + domain hints | [HF](https://huggingface.co/jk200201) |

Both are LoRA adapters for `Qwen/Qwen2.5-Coder-7B-Instruct`. Spider training data: [spider-dpo-1040](https://huggingface.co/datasets/jk200201/spider-dpo-1040).

---

## How it was trained (the interesting part)

Instead of human-annotated preferences, the training signal comes from **disagreements between two frontier models**:

```
Grok-4      ─┐
             ├─► run on Spider ─► where one is right and the other wrong,
DeepSeek-V3 ─┘   the correct SQL is "chosen", the wrong one "rejected"
                              │
                              ▼
            SFT on gold SQL  →  DPO on the preference pairs  →  7B that beats both
```

- **SFT**: QLoRA (4-bit NF4, rank 32, α 64), 3 epochs, LR 2e-4 cosine
- **DPO**: 1,040 clear-preference pairs, β 0.1, 2 epochs, LR 5e-5 cosine
- **Cost**: ~$25 of API calls · **Hardware**: one A10G (24 GB) · **Time**: ~3h

The pipeline is benchmark-agnostic (baseline eval → preference pairs → LLM judge → DPO) and also runs on BIRD. Full method, BIRD experiments, and the cost breakdown are in [CLAUDE.md](CLAUDE.md).

## Limitations

- English questions, SQLite dialect (Postgres/MySQL connectors are on the roadmap).
- Large real-world schemas (100s of tables) exceed the context window — schema-linking/retrieval is the next milestone.
- 4-bit trades a little accuracy for memory; use `--bf16` for best results.

## Roadmap

- [ ] Postgres / MySQL connectors + auto schema introspection
- [ ] Schema linking / retrieval for large databases
- [ ] Hosted demo (Hugging Face Space)
- [ ] "High-accuracy mode" (best-of-N self-consistency)

## Citation

```bibtex
@misc{kothari2026localsql,
  author = {Kothari, Jenish},
  title  = {A 7B Text-to-SQL Model that Competes with Frontier Models via Frontier-Disagreement DPO},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo}},
}
```
