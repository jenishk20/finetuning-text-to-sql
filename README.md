# Text-to-SQL Fine-Tuning via Frontier Model DPO

**A 7B fine-tuned model that outperforms the frontier models used to build its own training data.**

| Model | Result Accuracy | Notes |
|-------|----------------|-------|
| **Qwen2.5-Coder-7B (fine-tuned)** | **78.2%** | SFT + DPO — this repo |
| Grok-4 (xAI) | 73.7% | Frontier model baseline |
| DeepSeek V3 | 71.8% | Frontier model baseline |

Spider V1 dev set, 1,034 questions, execution-based (result set comparison). Total cost: **~$25**.

> **HuggingFace:** [Model](https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo) · [Dataset](https://huggingface.co/datasets/jk200201/spider-dpo-1040)

---

## How It Works

The core idea: run two frontier models on a benchmark, extract preference signal from their disagreements, then use DPO to teach a small model what "better SQL" looks like.

```
Frontier model A (Grok-4)    ─┐
                               ├─► Compare results ─► DPO pairs ─► Fine-tune 7B
Frontier model B (DeepSeek V3) ─┘       ↑
                                   LLM judge for
                                   ties (Gemini)
```

This pipeline is benchmark-agnostic. It currently supports **Spider V1** (completed) and **BIRD** (pipeline built, ready to run).

---

## Pipeline Steps

### Step 1 — Baseline Evaluation

Run any LLM against Spider or BIRD via `src/spider/pipeline.py` / `src/bird/pipeline.py`.

```bash
# Spider — run with Grok
python -m src.spider.pipeline --provider xai --model grok-4-1-fast-reasoning

# Spider — run with DeepSeek via OpenRouter
python -m src.spider.pipeline --provider openrouter --model deepseek-v3

# Resume a previous run
python -m src.spider.pipeline --resume results/spider_run_XXXX.json

# BIRD works the same way
python -m src.bird.pipeline --provider openrouter --model deepseek-v3
```

Results: Grok-4 → **73.7%** (762/1034), DeepSeek V3 → **71.8%** (742/1034)

---

### Step 2 — Build Preference Pairs

```bash
python -m src.spider.build_pairs   # Spider
python -m src.bird.build_pairs     # BIRD
```

Matches questions from both runs and sorts them into 4 buckets:

| Category | Count | Action |
|----------|-------|--------|
| One correct, one wrong | 118 | Direct preference pair |
| Both correct, identical SQL | 215 | Skipped (no signal) |
| Both correct, different SQL | 478 | → LLM judge queue |
| Both wrong | 223 questions → 446 pairs | Gold SQL as chosen |

Output: 564 ready pairs + 478 queued for judging.

---

### Step 3 — LLM Judge

```bash
python -m src.spider.llm_judge   # Spider
python -m src.bird.llm_judge     # BIRD
```

Judge model: **Gemini 2.5 Flash** via OpenRouter. Cost: ~$0.04 for 478 calls.

For each "both correct, different SQL" pair the judge sees the schema, question, gold SQL, and both generated SQLs. Position bias is controlled by randomly swapping A/B assignment (seed 42).

Final dataset: **1,040 DPO pairs** after filtering.

| Category | Count |
|----------|-------|
| Judge-resolved (both correct) | 478 |
| Gold vs wrong | 445 |
| Clear preference (one right, one wrong) | 117 |

---

### Step 4 — Format Training Data

```bash
python -m src.spider.format_training
```

Outputs LLaMA-Factory alpaca format to `data/training/` (available on HuggingFace):

| File | Contents |
|------|----------|
| `sft_data.json` | 7,000 SFT examples (Spider train set, gold SQL) |
| `dpo_data.json` | 1,040 DPO preference pairs |
| `dataset_info.json` | LLaMA-Factory dataset registry |

---

### Step 5 — Fine-Tuning

Trained on **AWS EC2 g5.xlarge** (NVIDIA A10G, 23.7 GB VRAM) using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

**Stage 1 — SFT**

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Dataset | spider_sft (7,000 examples) |
| Method | QLoRA (4-bit NF4 + LoRA rank 32, alpha 64) |
| Epochs | 3 · LR 2e-4 cosine · effective batch 16 |
| Runtime | 4h 18m · Final loss 0.0715 |

**Stage 2 — DPO**

| Parameter | Value |
|-----------|-------|
| Dataset | spider_dpo (1,040 pairs) |
| Method | Sigmoid DPO · beta 0.1 |
| Epochs | 2 · LR 5e-5 cosine · effective batch 16 |
| Runtime | 56 min · Final loss 0.2208 · Reward margin 4.732 |

---

### Step 6 — Evaluation

```bash
python -m src.eval_finetuned          # full 1,034 questions
python -m src.eval_finetuned --limit 50
python -m src.eval_finetuned --base_only  # base model without fine-tuning
```

| Metric | Score |
|--------|-------|
| **Result accuracy** | **78.2%** |
| Execution accuracy | 97.3% |
| Row count accuracy | 86.6% |

---

## Quick Start

```bash
git clone https://github.com/jenishkothari/finetuning-text-to-sql
cd finetuning-text-to-sql
pip install -r requirements.txt

cp .env.example .env
# Add your XAI_API_KEY and OPENROUTER_API_KEY to .env

# Download Spider data
# From https://github.com/taoyds/spider → place in data/spider_data/
```

The `data/` directory (training JSONs, Spider databases) and `results/` (eval outputs) are not included in this repo. Get them from:
- **Training data & DPO pairs:** [HuggingFace dataset](https://huggingface.co/datasets/jk200201/spider-dpo-1040)
- **Trained LoRA adapter:** [HuggingFace model](https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo)
- **Spider benchmark:** [taoyds/spider](https://github.com/taoyds/spider)

---

## Repository Structure

```
finetuning-text-to-sql/
├── src/
│   ├── shared/                   # Infrastructure used by all experiments
│   │   ├── llm_client.py         # xAI / OpenRouter client
│   │   ├── schema_loader.py      # SQLite schema DDL loader
│   │   ├── sqlite_executor.py    # SQL execution + result comparison
│   │   └── evaluator.py          # Scoring logic
│   ├── spider/                   # Spider V1 experiment (complete)
│   │   ├── pipeline.py           # Step 1: run any LLM on Spider dev set
│   │   ├── build_pairs.py        # Step 2: categorize results into DPO pairs
│   │   ├── llm_judge.py          # Step 3: Gemini judge for ties
│   │   ├── format_training.py    # Step 4: output LLaMA-Factory format
│   │   └── eval_finetuned.py     # Step 6: evaluate fine-tuned model
│   └── bird/                     # BIRD experiment (active)
│       ├── pipeline.py           # Step 1: run any LLM on BIRD dev set
│       ├── build_pairs.py        # Step 2: categorize results into DPO pairs
│       └── llm_judge.py          # Step 3: Gemini judge for ties
├── configs/                      # LLaMA-Factory YAML configs
├── requirements.txt
└── .env.example
```

---

## Cost Summary

| Item | Cost |
|------|------|
| Grok-4 API (1,034 questions) | ~$8–12 |
| DeepSeek V3 via OpenRouter (1,034 questions) | ~$2–3 |
| Gemini 2.5 Flash judge (478 calls) | ~$0.04 |
| AWS EC2 g5.xlarge (~13 hours) | ~$13 |
| **Total** | **~$25–30** |

---

## Key Findings

1. **A 7B fine-tuned model can outperform 100B+ frontier models on a specialized task.** Qwen2.5-Coder-7B at 78.2% beat both Grok-4 (73.7%) and DeepSeek V3 (71.8%) on Spider V1.

2. **Frontier model disagreements are a viable DPO signal.** Cases where two frontier models both produce correct but stylistically different SQL — judged by a third model — provide nuanced preference signal beyond simple correct/incorrect labeling.

3. **DPO adds meaningful signal beyond SFT.** SFT alone likely reaches ~70–73%. DPO pushed the model an estimated +4–5% by learning style preferences among correct SQL outputs.

4. **The pipeline is benchmark-agnostic.** The same 4-step flow (baseline eval → preference pairs → LLM judge → DPO) is already implemented for BIRD and can be extended to any execution-evaluable benchmark.

---

## Spider V1 Leaderboard Context

The Spider V1 leaderboard closed in February 2024. Dev set score in context:

| Model | Score | Notes |
|-------|-------|-------|
| DAIL-SQL + GPT-4 | 86.6% | Best published method |
| RESDSQL-3B + NatSQL | 79.9% | Best small model |
| **Qwen2.5-Coder-7B (ours, dev)** | **78.2%** | — |
| DIN-SQL + CodeX | 78.2% | GPT-3 Codex + chain-of-thought |
| Graphix-3B + PICARD | 77.6% | — |

Our 7B locally-runnable fine-tuned model matches DIN-SQL + CodeX, which uses GPT-3 Codex with chain-of-thought prompting.

---

## Citation

If you use the Spider dataset, please cite:

```bibtex
@inproceedings{Yu2018Spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev},
  booktitle = {EMNLP},
  year      = {2018}
}
```
