# Text-to-SQL on a Small Model: What Worked and What Did Not

An execution-verified study of post-training methods for text-to-SQL on Qwen2.5-Coder-7B and 14B, benchmarked on BIRD and Spider V1.

The short version: **reasoning distillation and inference-time sampling worked. Every preference-optimization and RL method we tried did not.** A 7B model reaches 58.5% on BIRD dev with self-consistency, which matches DeepSeek V4-Pro (58.7%) at roughly 0.4% of the parameters. All numbers below are execution-based result accuracy, measured by running generated SQL against the real databases and comparing result sets.

---

## Correction to an earlier result

An earlier version of this repository reported **78.2% on Spider V1 dev** and claimed the fine-tuned 7B "outperforms the frontier models used to build its own training data." **That claim is retired.** It had train/test contamination in the DPO stage: preference pairs were built from predictions on the Spider **dev** set, and the model was then evaluated on that same dev set. 445 of the 1,040 pairs used the dev question's own gold SQL as the chosen response.

The clean re-run tells a different and more interesting story:

| Spider V1 dev (1,034 Q) | Result accuracy |
|---|---|
| Base Qwen2.5-Coder-7B-Instruct, no adapter | 77.2% |
| SFT on Spider train (7,000 gold) | 77.0% |
| SFT on 5,500 split | 77.1% |
| SFT + GRPO on held-out 1,500 | 77.3% |
| Retired contaminated SFT + DPO | ~~78.2%~~ |

The SFT stage was always clean (trained on `train_spider.json`, disjoint databases). Only DPO leaked. But the more important finding is the first row: **the base model with no fine-tuning at all scores 77.2%.** Spider V1 is saturated for a modern coding-specialist 7B, and the measured lift from fine-tuning is approximately zero. The original 78.2% was a strong base model plus about one point of contamination.

For reference, on the same strict metric: Qwen3-Coder-480B 78.8%, DeepSeek-V4-Pro 76.2%, DeepSeek-V3.1 74.8%, Grok-4 73.7%, GLM-5.2 67.0%. The untuned 7B beats most of them out of the box, which says more about Spider V1 as a benchmark than about any method in this repo.

BIRD experiments were built correctly throughout (pairs from train, evaluation on dev) and are unaffected.

---

## BIRD results

BIRD dev, 1,534 questions, Qwen2.5-Coder-7B-Instruct unless noted.

| Stage | Result accuracy |
|---|---|
| Base model, no adapter | 27.0% |
| SFT on BIRD train, direct SQL | 46.9% |
| SFT + frontier DPO, 1,219 correctness pairs | 50.3% |
| **CoT-SFT** (reasoning distilled from Qwen3-Coder-480B) | **52.1%** |
| **CoT-SFT + Best-of-N, K=8, self-consistency** | **58.5%** |
| 14B SFT, direct SQL | 55.0% |
| 14B SFT + Best-of-N, K=4 | 59.3% |

Two levers moved the model. Distilling the teacher's **reasoning** rather than its SQL answers generalized better across BIRD's unseen dev databases, worth +5.2pp over direct-SQL SFT. Then **Best-of-N with execution-based majority voting** added a further +6.4pp at inference time with no retraining. Selecting the first executable candidate instead of majority-voting recovers only +0.9pp, so the voting is doing the work, not the extra samples.

### Frontier comparison

Measured locally on the full 1,534-question dev set with chain-of-thought prompting for every model, single sample:

| Model | Params | Result accuracy |
|---|---|---|
| GLM 5.2 | 744B | 63.0% |
| DeepSeek V4-Pro | 1.6T | 58.7% |
| **This work, 7B + Best-of-N K=8** | **7B** | **58.5%** |
| This work, 7B greedy CoT | 7B | 52.1% |

Chain-of-thought prompting is used for all models deliberately. Direct-SQL prompting handicaps reasoning models badly (GLM 5.2 drops to 47.6%), so direct-SQL comparisons would flatter this work and are not reported. The honest claim is **parameter efficiency, not superiority**: the 7B ties DeepSeek V4-Pro at a fraction of the size and runs locally, while GLM 5.2 wins outright.

---

## Negative results

This is the most useful part of the repository. Every method below was run to completion with execution-based evaluation. All deltas are against the appropriate SFT baseline on the same benchmark.

| Method | Model | Outcome | Delta |
|---|---|---|---|
| Off-policy DPO, correctness pairs only (1,219) | 7B | 50.3% | **+3.4** |
| Off-policy DPO, same 1,219 pairs | 14B | 49.3% | −5.7 |
| DPO with judge-resolved style pairs (4,677) | 7B | 40.7% | −9.6 |
| Delta-learning DPO (8,293 pairs) | 7B | 35.3% | −15.0 |
| Self-sampling DPO (680 pairs) then Best-of-N | 14B | 58.7% | −0.6 |
| On-policy CoT-DPO (514 pairs) | 7B | regressed | −8.2 |
| STaR / rejection-sampling FT (1,664 self CoT) | 7B | degenerated | ~−20 |
| GRPO, binary execution reward, from SFT | 7B, BIRD | 46.9% | 0.0 |
| GRPO, binary execution reward, held-out split | 7B, Spider | 77.3% | +0.2 |

Four mechanisms explain the failures, and each is measurable:

**1. The annotator must be meaningfully stronger than the student.** The same 1,219 preference pairs that lifted the 7B by +3.4pp regressed the 14B by −5.7pp. The 14B base (54.3%) already matched the frontier models that produced the "chosen" responses, so the preference signal became noise. This is a controlled natural experiment: identical data, identical procedure, opposite sign, explained entirely by the student-teacher gap.

**2. Style preferences actively hurt a correctness metric.** Adding 3,458 judge-resolved pairs (both models correct, different SQL) on top of 1,219 correctness pairs cost 9.6pp. BIRD scores result sets, not SQL aesthetics, so style pairs carry zero signal and dilute the gradient.

**3. Preference optimization collapses the diversity that Best-of-N depends on.** Self-sampling DPO on the 14B produced a healthy training curve (rewards/accuracies 47.5% to 71%, margins +0.5) and still made the model worse downstream. The mechanism is visible in the pick distribution: the top-ranked sample's win rate rose from 79.4% to 87.2%. The greedy output barely improved while the alternatives stopped being useful, and the damage concentrated on challenging questions (−4.1pp) where exploration matters most. **If you plan to sample at inference time, preference optimization can be actively counterproductive.**

**4. Binary execution reward saturates, and a leaky holdout hides it.** GRPO on the 7,000 Spider training questions the model had already memorized produced reward pinned at 0.98 with std ~0.02, so there was no gradient signal and the run was a no-op. Re-splitting into a disjoint 5,500/1,500 partition restored reward variance (~0.88) and genuine signal, and still yielded only +0.2pp, because the 1,500 holdout questions were drawn from databases the SFT stage had already seen while dev uses unseen databases. Spider pass@8 is 84.7% against 77.3% greedy, so real headroom exists and neither DPO nor GRPO harvested any of it.

The conclusion we draw: on execution-verified text-to-SQL at this scale, **inference-time compute beat every training-time method we tried**, and the two-line summary of the training work is that reasoning distillation helps and preference optimization is a trap unless the annotator gap is large.

---

## Artifacts

| Artifact | Link |
|---|---|
| Merged model, BIRD CoT-SFT | [jk200201/qwen2.5-coder-7b-bird-cot](https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-cot) |
| LoRA adapter | [jk200201/qwen2.5-coder-7b-bird-cot-lora](https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-cot-lora) |
| GGUF quants (Q4_K_M / Q5_K_M / Q8_0 / f16) | [jk200201/qwen2.5-coder-7b-bird-cot-GGUF](https://huggingface.co/jk200201/qwen2.5-coder-7b-bird-cot-GGUF) |
| Ollama | `ollama run muence/bird-cot` |
| Dataset, 5,593 execution-verified CoT examples | [jk200201/bird-cot-sft](https://huggingface.co/datasets/jk200201/bird-cot-sft) |
| Live demo | [jk200201/text2sql-live](https://huggingface.co/spaces/jk200201/text2sql-live) |

The [spider-dpo-1040](https://huggingface.co/datasets/jk200201/spider-dpo-1040) dataset remains available but carries a contamination warning on its card. Do not train on its `dpo` subset and then report Spider dev as a clean held-out split.

---

## Pipeline

The repository implements a benchmark-agnostic flow over any execution-evaluable text-to-SQL dataset. Spider V1 and BIRD are both wired up.

### 1. Baseline evaluation

Run any model against a benchmark and score it by execution.

```bash
python -m src.bird.pipeline --provider wandb --model glm-5.2 --cot --workers 8
```

Providers: `xai`, `openrouter`, `wandb`, `llamacpp`. Use `--list-models` to enumerate. `--cot` is required for a fair comparison against reasoning models.

### 2. Build preference pairs

```bash
python -m src.bird.build_pairs
```

Sorts matched questions into four buckets: one correct and one wrong (usable), both correct with identical SQL (no signal), both correct with different SQL (style only, **do not use**, see negative result 2), and both wrong.

### 3. Build the CoT training set

```bash
python -m src.bird.build_cot_sft_data
```

A teacher generates reasoning chains for each training question, the final SQL from each chain is executed, and only chains producing correct results are kept. Yield was 59% (5,593 of 9,428) with Qwen3-Coder-480B as teacher, at roughly $8 of inference credits.

### 4. Train

```bash
python -m src.bird.sft_train      # SFT and CoT-SFT (QLoRA)
python -m src.bird.dpo_train      # DPO
python -m src.bird.grpo_train     # GRPO, binary execution reward
```

CoT-SFT trains a fresh LoRA on the base model, not on top of the direct-SQL adapter. Two epochs, cutoff length 8192. Truncating schemas below that cost measurable accuracy.

### 5. Evaluate

```bash
python -m src.bird.eval_finetuned --cot            # greedy
python -m src.bird.eval_best_of_n --k 8 --cot      # self-consistency, vLLM
```

Best-of-N selects by execution-based majority vote over result sets, which is what produces the gain.

---

## Reproduction notes

Hard-won details that cost real time. Recorded so they cost you less.

**Environment.** One working combination for the whole pipeline: `transformers 4.46.3` + `trl 0.12.2` + `vllm 0.6.3`. Newer TRL or transformers break vLLM 0.6.3; older versions break TRL. GRPO needs `trl>=0.14`, so it gets its own environment with **vLLM excluded**, since vLLM's pins break TRL.

**Split GPU generation from CPU execution into separate jobs.** vLLM generates all candidates, then SQL execution runs for hours on CPU while the GPU sits idle. On a managed cluster an idle-GPU watchdog will kill the job.

**Evaluate on unseen databases.** Both benchmarks use disjoint train/dev databases by design, and that gap is where the difficulty lives. A holdout carved from training questions on training databases will show reward signal and transfer nothing.

**GRPO loss reads ~0 by design.** Group-relative advantage centers it. Judge health by rising reward and non-zero grad norm.

**Best-of-N needs majority voting.** First-executable selection gives +0.9pp; voting gives the full +6.4pp.

---

## Repository structure

```
src/
├── shared/                    Infrastructure shared by all experiments
│   ├── llm_client.py          Multi-provider client (xai, openrouter, wandb, llamacpp)
│   ├── schema_loader.py       SQLite schema DDL loader
│   ├── sqlite_executor.py     Sandboxed execution + result-set comparison
│   └── evaluator.py           Scoring
├── bird/                      BIRD experiments
│   ├── pipeline.py            Baseline evaluation
│   ├── build_pairs.py         Preference pair construction
│   ├── build_cot_sft_data.py  Teacher CoT generation + execution filter
│   ├── build_self_sampling_pairs.py
│   ├── sft_train.py  dpo_train.py  grpo_train.py
│   ├── inference.py           vLLM engine, CoT prompting, SQL extraction
│   ├── eval_finetuned.py      Greedy evaluation
│   └── eval_best_of_n.py      Self-consistency evaluation
└── spider/                    Spider V1 experiments
scripts/                       SLURM job scripts
configs/                       LLaMA-Factory YAML
model_cards/                   Published HuggingFace cards
```

## Setup

```bash
git clone https://github.com/jenishk20/finetuning-text-to-sql
cd finetuning-text-to-sql
pip install -r requirements.txt
cp .env.example .env
```

Benchmark data is not included. Get Spider from [taoyds/spider](https://github.com/taoyds/spider) into `data/spider_data/`, and BIRD from [bird-bench.github.io](https://bird-bench.github.io/) into `data/bird_data/`.

## Citation

```bibtex
@inproceedings{Yu2018Spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and
               Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and
               Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and
               Yao, Qingning and Roman, Shanelle and Zhang, Zilin and Radev, Dragomir},
  booktitle = {EMNLP},
  year      = {2018}
}

@article{li2023bird,
  title   = {Can LLM Already Serve as A Database Interface? A Big Bench for
             Large-Scale Database Grounded Text-to-SQLs},
  author  = {Li, Jinyang and Hui, Binyuan and Qu, Ge and Yang, Jiaxi and
             Li, Binhua and Li, Bowen and Wang, Bailin and others},
  journal = {NeurIPS},
  year    = {2023}
}
```
