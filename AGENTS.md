# AGENTS.md — Text-to-SQL Fine-Tuning Experiments (Spider + BIRD)

## Project Summary

A research experiment that fine-tunes **Qwen2.5-Coder-7B-Instruct** using a novel DPO pipeline: run two frontier models (Grok-4 and DeepSeek V3) against a benchmark, extract preference signal from their disagreements, then use DPO to train a 7B model that beats both.

**Results:**
- **Spider V1**: Fine-tuned 7B scores **78.2%** — beats Grok-4 (73.7%) and DeepSeek V3 (71.8%)
- **BIRD**: Fine-tuned 7B scores **46.9%** — below frontier models (~55%), +20pp over base model (27%)

HuggingFace: [model](https://huggingface.co/jk200201/qwen2.5-coder-7b-sql-dpo) · [dataset](https://huggingface.co/datasets/jk200201/spider-dpo-1040)

---

## Tech Stack

- **Language**: Python 3.13
- **LLM APIs**: OpenAI-compatible client (`openai` SDK) for xAI (Grok) and OpenRouter (DeepSeek, Gemini)
- **Fine-tuning framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (run externally on AWS EC2)
- **Local inference**: `transformers` + `peft` + `bitsandbytes` (4-bit QLoRA)
- **Databases**: SQLite (benchmark DBs accessed directly via `sqlite3`)
- **Config**: `python-dotenv` + `pyyaml`
- **Training hardware**: AWS EC2 g5.xlarge (NVIDIA A10G 24GB VRAM)

---

## Repository Structure

```
finetuning-text-to-sql/
├── src/
│   ├── shared/                   # Infrastructure used by all experiments
│   │   ├── llm_client.py         # Unified xAI/OpenRouter client — all LLM calls go here
│   │   ├── schema_loader.py      # Load SQLite schemas as DDL strings
│   │   ├── sqlite_executor.py    # Execute SQL + return result sets
│   │   └── evaluator.py          # compare_results(), compute_metrics()
│   │
│   ├── spider/                   # Spider V1 experiment (complete, published)
│   │   ├── pipeline.py           # Step 1: run LLM on Spider dev set (1,034 Qs)
│   │   ├── build_pairs.py        # Step 2: categorize results into DPO pairs
│   │   ├── llm_judge.py          # Step 3: Gemini judge for ties
│   │   ├── format_training.py    # Step 4: output LLaMA-Factory alpaca/pairwise format
│   │   ├── eval_finetuned.py     # Step 6: evaluate fine-tuned model locally
│   │   ├── analyze_results.py    # Analyze failure patterns by DB/complexity
│   │   └── finetune.py           # Early SFT-only baseline (Gretel dataset, deprecated)
│   │
│   └── bird/                     # BIRD experiment (active)
│       ├── pipeline.py           # Step 1: run LLM on BIRD dev set (1,534 Qs)
│       ├── build_pairs.py        # Step 2: categorize results into DPO pairs
│       └── llm_judge.py          # Step 3: Gemini judge for ties
│                                 # (delta learning scripts go here on bird-delta-learning branch)
│
├── configs/config.yaml           # Model provider + evaluation settings
├── data/
│   ├── bird_data/                # BIRD dev set (dev.json, dev_databases/)
│   ├── spider_data/              # Spider benchmark (not in repo — download separately)
│   └── training/                 # LLaMA-Factory formatted data (sft_data.json, dpo_data.json)
├── models/
│   ├── bird_dpo/                 # BIRD DPO LoRA adapter (local)
│   └── qwen-7b-sql-dpo/          # Spider DPO LoRA adapter (also on HuggingFace)
├── results/                      # Pipeline output JSONs (timestamped), preference pairs
└── .env                          # API keys (XAI_API_KEY, OPENROUTER_API_KEY)
```

---

## The 6-Step Pipeline

### Step 1 — Baseline Evaluation
```bash
python -m src.spider.pipeline --provider xai --model grok-4-1-fast-reasoning
python -m src.spider.pipeline --provider openrouter --model deepseek-v3
python -m src.bird.pipeline --provider openrouter --model deepseek-v3
python -m src.spider.pipeline --resume results/spider_run_XXXX.json   # resume interrupted run
```
Saves timestamped JSON to `results/`. Checkpoints every 25 questions.

### Step 2 — Build Preference Pairs
```bash
python -m src.spider.build_pairs   # Spider → results/preference_pairs.json + judge_queue.json
python -m src.bird.build_pairs     # BIRD   → same structure
```
4 categories: `clear_preference` (1 right, 1 wrong) · `judge_needed` (both correct, different SQL) · `gold_vs_wrong` (both wrong, use gold SQL) · `skip` (both correct, identical SQL)

### Step 3 — LLM Judge
```bash
python -m src.spider.llm_judge   # Spider → results/preference_pairs_final.json
python -m src.bird.llm_judge     # BIRD
```
Judge: Gemini 2.5 Flash via OpenRouter. Position bias controlled by random A/B swap (seed 42). ~$0.04–0.05 total.

### Step 4 — Format Training Data
```bash
python -m src.spider.format_training   # → data/training/sft_data.json + dpo_data.json
```
LLaMA-Factory alpaca format. SFT uses Spider train gold SQL (7,000 examples). DPO uses preference pairs (1,040 Spider / 1,914 BIRD).

### Step 5 — Fine-Tuning (external, on AWS EC2)
Done via LLaMA-Factory YAML configs. Two stages:
1. **SFT**: QLoRA (4-bit NF4, LoRA rank 32, alpha 64), 3 epochs, LR 2e-4 cosine
2. **DPO**: Sigmoid DPO, beta 0.1, 2 epochs, LR 5e-5 cosine

### Step 6 — Evaluation
```bash
python -m src.spider.eval_finetuned               # full Spider dev set (local model)
python -m src.spider.eval_finetuned --limit 50    # quick smoke test
python -m src.spider.eval_finetuned --base_only   # base model without LoRA
```
BIRD evaluation uses a separate script (`~/bird_eval.py` on the EC2 instance).

---

## Key Conventions

**Running scripts**: Always run as `python -m src.<script>` from the project root — never as `python src/script.py`.

**LLM calls**: All go through `src/llm_client.py:call_llm()`. Providers are `xai` and `openrouter`. Model shortcuts resolve via `OPENROUTER_MODELS` dict. Temperature 0.0 for deterministic SQL.

**SQL extraction**: `extract_sql()` in `llm_client.py` strips markdown fences and `<think>...</think>` blocks (for DeepSeek R1).

**Instruction template** (must match between training and inference):
```
Convert the following natural language question into a valid SQL query.

Database Schema:
{schema}

Question: {question}

Return only the SQL query with no explanation.
```

**Evaluation metric**: "Result accuracy" = result set comparison (not string match). A query passes if it returns the same rows as the gold SQL.

**BIRD vs Spider differences**: BIRD questions have an `evidence` field (domain knowledge) injected into the prompt. Gold SQL field is `"SQL"` (not `"query"`).

**Result files**: Named `{benchmark}_{model}_{timestamp}.json`. Final preference pair files: `preference_pairs_final.json` (Spider), `bird_preference_pairs_final.json` (BIRD).

---

## API Keys (.env)
```
XAI_API_KEY=...          # Grok-4 via api.x.ai
OPENROUTER_API_KEY=...   # DeepSeek, Gemini Flash, etc.
```

---

## Key Results Reference

| Benchmark | Model | Accuracy |
|-----------|-------|----------|
| Spider V1 | Qwen2.5-Coder-7B **fine-tuned** | **78.2%** |
| Spider V1 | Grok-4 (frontier baseline) | 73.7% |
| Spider V1 | DeepSeek V3 (frontier baseline) | 71.8% |
| BIRD | Grok-4 | 55.4% |
| BIRD | DeepSeek V3 | 54.7% |
| BIRD | Qwen2.5-Coder-7B **fine-tuned** | 46.9% |
| BIRD | Qwen2.5-Coder-7B base | 27.0% |

**BIRD gap analysis** (see `BIRD_RESULTS.md`): Likely causes are schema truncation (`cutoff_len: 1536` truncated 15.4% of examples), cross-domain generalization (train/dev use different databases), and `evidence` field complexity.

---

## What's Not in the Repo

- `data/spider_data/` — download from [taoyds/spider](https://github.com/taoyds/spider)
- `data/training/` — training JSONs on [HuggingFace dataset](https://huggingface.co/datasets/jk200201/spider-dpo-1040)
- Most `results/*.json` — large eval output files (gitignored except `.gitkeep`)

---

## Repo Organization Decision

**One repo, not two.** Spider (complete) and BIRD (active) share `llm_client.py`, `schema_loader.py`, `sqlite_executor.py`, and `evaluator.py`. Splitting would cause code duplication.

Planned reorganization:
```
src/
├── shared/    # llm_client, schema_loader, sqlite_executor, evaluator
├── spider/    # all Spider scripts (complete, published)
└── bird/      # all BIRD scripts (active experiment)
```

Use a branch (`bird-delta-learning`) for the next experiment rather than a new repo.

---

## Next Experiment: Delta Learning on BIRD

**Paper**: "The Delta Learning Hypothesis" (Geng et al., COLM 2025) — arXiv:2507.06187

**Core idea**: Preference pairs where both chosen and rejected are weak (from small models) can improve a stronger student model via DPO, because the *relative quality delta* drives learning even when absolute quality is low. SFT on the same weak data *hurts*; DPO on the pairs *helps*.

### Proposed Pipeline

| Step | Action |
|------|--------|
| 1 | Run **Qwen2.5-Coder-3B-Instruct** on BIRD **train set** → get SQL outputs |
| 2 | Run **Qwen2.5-Coder-1.5B-Instruct** on BIRD **train set** → get SQL outputs |
| 3 | Build preference pairs (see logic below) |
| 4 | SFT Qwen 7B on BIRD train gold SQL (already done: `bird_sft_data.json`) |
| 5 | DPO on delta learning pairs |
| 6 | Evaluate on BIRD dev set |

**Important**: Run steps 1–2 on the **BIRD train set**, not the dev set (dev is the evaluation split).

### Preference Pair Logic

```python
if qwen3b_correct and not qwen1b_correct:
    # Clean delta pair — strongest signal
    chosen = qwen3b_sql
    rejected = qwen1b_sql

elif qwen3b_correct and qwen1b_correct and not sqls_equivalent:
    # Both correct, different SQL — size as quality proxy
    chosen = qwen3b_sql
    rejected = qwen1b_sql

elif qwen3b_correct and qwen1b_correct and sqls_equivalent:
    pass  # skip — no delta

elif not qwen3b_correct and not qwen1b_correct:
    # Both wrong — gold SQL fallback (max delta, still valid)
    chosen = gold_sql
    rejected = qwen1b_sql  # or both as separate pairs

elif qwen1b_correct and not qwen3b_correct:
    pass  # inverted quality — skip
```

**Why gold SQL fallback when both wrong?** On BIRD, SQL quality is binary (correct/incorrect), not a spectrum like general instruction following. Qwen 3B likely gets only ~15-20% right on BIRD, Qwen 1.5B ~5-10%. ~40-50% of questions will have both models wrong. Gold SQL fallback gives maximum delta and is consistent with the paper's spirit — the delta is directionally correct even if one party is externally sourced.

### Realistic Accuracy Forecast

| Improvement | Expected gain | Cumulative |
|------------|--------------|-----------|
| Current (SFT + frontier DPO) | — | 46.9% |
| + Delta learning DPO | +2–4pp | ~49–51% |
| + cutoff_len 4096 (g5.2xlarge, 48GB) | +3–5pp | ~52–56% |
| + Schema pruning | +3–5pp | ~55–61% |

**Delta learning alone will not reach 55-60%.** The primary BIRD bottleneck is schema truncation (cutoff_len 1536 truncated 15.4% of training examples). Must stack delta learning with a larger GPU and schema pruning to hit that range.

### Key Dependency: SFT Checkpoint

DPO must be applied on top of the **SFT checkpoint**, not the already-DPO'd model. Check whether the intermediate SFT checkpoint is still on the EC2/EBS volume before running the delta learning DPO. If not, re-run SFT first using `bird_sft_data.json` (9,428 examples, already formatted).

### Scripts Built (bird-delta-learning branch)
- `src/bird/local_pipeline.py` — run Qwen 3B/1.5B locally on BIRD train set with SQL execution
- `src/bird/build_bird_delta_pairs.py` — build preference pairs with gold SQL fallback
- `~/format_delta_pairs.py` (EC2 only) — format pairs for LLaMA-Factory DPO

---

## Delta Learning Experiment Results (Attempt 1 — FAILED)

**Date**: April 2026 | **Branch**: `bird-delta-learning`

### What Was Run

| Step | Detail |
|------|--------|
| Qwen 3B on BIRD train (9,428 Qs) | 39.6% accuracy |
| Qwen 1.5B on BIRD train (6,000 Qs) | 27.1% accuracy |
| Preference pairs built | 8,293 total |
| DPO training config | cutoff_len 1200, beta 0.1, 2 epochs, on top of bird_sft |
| **Final dev accuracy** | **35.3%** (down from 46.9%) |

### Pair Breakdown
| Category | Questions | Pairs |
|----------|-----------|-------|
| clean_delta (3B right, 1.5B wrong) | 1,195 | 1,195 |
| both_correct (different SQL) | 736 | 736 |
| gold_fallback (both wrong) | 3,181 | **6,362** |
| skip_inverted | 386 | 0 |
| skip_identical | 502 | 0 |
| **Total** | | **8,283** |

### Why It Failed (root causes in order)

1. **Gold fallback pairs dominated (77% of training)** — when both models were wrong, gold SQL as chosen created 6,362 pairs. At 77% of training data this became the entire signal, not a fallback. The model learned to avoid weak SQL patterns but overcorrected, drifting far from the SFT base.

2. **DPO overfit** — loss dropped to 0.009 and rewards/accuracies hit 100% by epoch 1. Beta 0.1 with 2 epochs was too aggressive for this data mix.

3. **cutoff_len 1200** — needed to avoid OOM on A10G 24GB with DPO's double forward pass. Truncated ~25-30% of schemas vs 15% at 1536. Third factor, not primary cause.

### EC2 Checkpoints (all on EBS, safe)
- `~/checkpoints/bird_sft/` — SFT base (use this for next DPO attempt)
- `~/checkpoints/bird_dpo/` — frontier DPO at 46.9% (best result so far)
- `~/checkpoints/bird_delta_dpo/` — delta DPO at 35.3% (failed, can delete)
- `~/results/bird_delta_pairs.json` — 8,293 pairs (reuse for retry)

### Retry Plan (Attempt 2)

**Filter to clean pairs only:**
```bash
python3 -c "
import json
from pathlib import Path
pairs = json.load(open(Path.home() / 'results/bird_delta_pairs.json'))
clean = [p for p in pairs if p['category'] in ('clean_delta', 'both_correct')]
print(f'Filtered: {len(clean)} pairs (was 8283)')
json.dump(clean, open(Path.home() / 'results/bird_delta_pairs_clean.json', 'w'), indent=2)
"
# Produces ~1,931 high-quality pairs, no gold fallback
```

**Updated YAML changes:**
```yaml
pref_beta: 0.05        # was 0.1 — stay closer to SFT base
num_train_epochs: 1    # was 2 — stop before overfitting
cutoff_len: 1536       # restore original (may OOM — try gradient_checkpointing: true)
```

**Expected outcome**: ~47-50% (recover baseline + small gain from clean signal)

---

## Frontier DPO Experiment — BIRD Train Set (April 2026, In Progress)

**Branch**: `feat/bird-frontier-dpo`

### What Was Done

Ran two frontier models on the full BIRD **train set** (9,428 questions) via OpenRouter to build high-quality DPO preference pairs. Both models routed through OpenRouter (no separate xAI provider needed).

| Model | Shortcut | Result Accuracy | Exec Accuracy | Questions |
|-------|----------|----------------|---------------|-----------|
| DeepSeek V3.2 (`deepseek/deepseek-v3.2`) | `deepseek-v3` | **53.2%** | 93.1% | 9,428/9,428 ✓ |
| Grok 4.1 Fast (`x-ai/grok-4.1-fast`) | `grok-4.1-fast` | **53.5%** | 93.2% | 9,428/9,428 ✓ |

Both runs complete. Results saved on HPC at:
- Grok: `/scratch/phalle.y/results_frontier_bird_dpo_grok/bird_train_grok-4-1-fast_20260416_123222.json`
- DeepSeek: `/scratch/phalle.y/results_frontier_bird_dpo_deepseek/` (use `ls -t | head -1` for latest)

### HPC Environment

- **Cluster**: Northeastern Discovery (Apptainer container)
- **Python**: 3.8 (system) — all scripts need `from __future__ import annotations`
- **Virtual env**: `/scratch/phalle.y/finetuning-text-to-sql/myenv/`
- **Repo**: `/scratch/phalle.y/finetuning-text-to-sql/`
- **Train data**: `/scratch/phalle.y/bird_train/train/train.json`
- **Train DBs**: `/scratch/phalle.y/bird_train/train/train_databases/train_databases/`
- **SFT adapter**: `/home/phalle.y/Jenish-DPO-GRPO/bird_sft_adapter_1`
- **Activate env**: `source /scratch/phalle.y/finetuning-text-to-sql/myenv/bin/activate`

**pip install quirks on HPC Python 3.8:**
```bash
pip install "openai==1.40.0"   # newer openai requires jiter>=0.10.0 which has no py3.8 wheel
pip install "httpx<0.28.0"     # httpx>=0.28 removed proxies arg used by openai 1.40
pip install python-dotenv pyyaml
```

### Code Changes Made This Session

**`src/bird/pipeline.py`**
- Added `from __future__ import annotations` (Python 3.8 compat)
- Added `--train` flag to run on train set instead of dev set
- Added `--train-json`, `--db-dir`, `--output-dir` overrides for HPC paths
- Added `question_id` fallback for train set (train.json has no question_id field)
- Fixed `_save_results` to use `output_dir_override` and `parents=True`

**`src/shared/llm_client.py`**
- Added `from __future__ import annotations`
- Updated `xai` default model: `grok-4.1-fast`
- Updated `openrouter` default: `deepseek/deepseek-v3.2`
- Added `grok-4.1-fast` and `grok-4.1` shortcuts to `OPENROUTER_MODELS`

**`src/bird/build_pairs.py`** — complete rewrite
- Full CLI: `--grok`, `--deepseek`, `--db-dir`, `--output-dir`, `--include-gold-fallback`, `--dry-run`
- Gold fallback **skipped by default** — use `--include-gold-fallback` only if needed
- `from __future__ import annotations`
- Outputs: `bird_preference_pairs.json`, `bird_judge_queue.json`, `bird_preference_pairs_stats.json`

**`src/bird/llm_judge.py`**
- Added `from __future__ import annotations`
- Replaced all hardcoded paths with `--pairs-dir` CLI arg
- Reads/writes all files relative to `--pairs-dir`

**`src/bird/grpo_train.py`** (on `feat/bird-grpo`, merged to main)
- GRPO training using TRL GRPOTrainer
- `reward_funcs=[sql_reward_fn]` (must be list)
- `--resume` flag for checkpoint continuation

**`src/bird/grpo_dataset.py`** (on `feat/bird-grpo`, merged to main)
- Builds HuggingFace Dataset for GRPOTrainer from BIRD train JSON

### Next Steps (Frontier DPO)

**Step 1 — Build preference pairs** (run on HPC login node, no GPU)
```bash
cd /scratch/phalle.y/finetuning-text-to-sql
source myenv/bin/activate

# First pull latest code
git pull origin feat/bird-frontier-dpo

GROK_FILE=$(ls -t /scratch/phalle.y/results_frontier_bird_dpo_grok/ | head -1)
DS_FILE=$(ls -t /scratch/phalle.y/results_frontier_bird_dpo_deepseek/ | head -1)

python -m src.bird.build_pairs \
    --grok     /scratch/phalle.y/results_frontier_bird_dpo_grok/$GROK_FILE \
    --deepseek /scratch/phalle.y/results_frontier_bird_dpo_deepseek/$DS_FILE \
    --db-dir   /scratch/phalle.y/bird_train/train/train_databases/train_databases \
    --output-dir /scratch/phalle.y/results_frontier_pairs
```

**Step 2 — LLM judge** (~700 both-correct-different pairs, ~$0.05, runs locally or on HPC)
```bash
python -m src.bird.llm_judge \
    --pairs-dir /scratch/phalle.y/results_frontier_pairs
```

**Step 3 — Format for LLaMA-Factory** (script to be written — same format as Spider DPO)
Final file: `/scratch/phalle.y/results_frontier_pairs/bird_preference_pairs_final.json`

**Step 4 — Run DPO on H200**
```yaml
# Key config differences from previous failed attempts:
cutoff_len: 4096        # H200 80GB handles this (was 1200-1536 on A10G)
pref_beta: 0.05         # stay close to SFT base (was 0.1)
num_train_epochs: 1     # stop before overfitting (was 2)
save_steps: 50          # frequent saves for 2-hour session limit
```
Start from: `bird_sft_adapter_1` (NOT the 46.9% DPO checkpoint)

**Step 5 — Evaluate on BIRD dev**
```bash
python -m src.bird.pipeline \
    --provider openrouter --model deepseek-v3
    # (or with local model using eval_finetuned equivalent)
```

### Expected Accuracy

| Configuration | Expected |
|--------------|---------|
| Current best (SFT + old frontier DPO) | 46.9% |
| + New frontier DPO, A10G (cutoff 1536) | ~50-54% |
| + New frontier DPO, **H200 (cutoff 4096)** | **~55-60%** |

The H200's 80GB VRAM eliminates schema truncation (was 15.4% truncated at cutoff 1536).
With cutoff 4096, schema truncation drops to <2% → primary BIRD bottleneck removed.

### Pair Count Estimates (from 53% accuracy on both models)

| Category | Count |
|----------|-------|
| Clear preference (one right, one wrong) | ~4,700 |
| Judge needed (both correct, different SQL) | ~700 |
| Both wrong (skipped — no gold fallback) | ~2,000 |
| Identical SQL (skipped) | ~1,000 |
| **Total usable pairs** | **~5,400** |

Gold fallback is intentionally skipped. In the failed delta learning attempt, gold fallback was 77% of training data and caused regression to 35.3%. With frontier models at 53%, both-wrong rate is ~22% — lower but still risky.
