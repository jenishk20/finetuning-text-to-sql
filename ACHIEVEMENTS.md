# ACHIEVEMENTS — Text-to-SQL Fine-Tuning (Spider + BIRD)

A running record of everything accomplished in this project: a research program that
fine-tunes small open models (Qwen2.5-Coder **7B** and **14B**) to do text-to-SQL, using
frontier-model disagreement, reasoning distillation, self-sampling, Best-of-N, and
execution-feedback RL. Compiled from the full git history (6 branches), all experiment
results, project memory, and the published artifacts.

> **One-line headline:** A fine-tuned **7B model beats frontier models on Spider (78.2%)**,
> and on the much harder BIRD benchmark a **7B reaches 58.5%** (CoT-SFT + Best-of-N) —
> beating Grok-4.1 (55.4%) and DeepSeek V3.2 (54.7%) — starting from a 27% base.

---

## 🏆 Headline Achievements

1. **Spider V1 — a 7B model that beats the frontier models used to build its training data.**
   Fine-tuned Qwen2.5-Coder-7B scores **78.2%** result accuracy, beating **Grok-4 (73.7%)**
   and **DeepSeek V3 (71.8%)**. Published on HuggingFace (model + dataset). Total cost ~$25.

2. **BIRD — a 7B model that beats the frontier on a benchmark where the base model scores 27%.**
   The best 7B configuration (**CoT-SFT + Best-of-N self-consistency, K=8**) reaches **58.5%**,
   beating **Grok-4.1-fast (55.4%)** and **DeepSeek V3.2 (54.7%)** — a **+31.5pp** climb from
   the 27% base, and competitive with a fine-tuned 14B (59.3%).

3. **A novel, benchmark-agnostic DPO recipe** that turns two frontier models' *disagreements*
   into preference signal (correctness pairs + LLM-judged style pairs), validated end-to-end
   on Spider and ported to BIRD.

4. **A library of rigorously-validated negative results** that map the boundaries of each
   method (when DPO helps vs. hurts, when teacher swaps are useless, when self-sampling
   collapses diversity). These are first-class achievements — each one closed off a dead end
   with data, not guesswork.

5. **A full reproducible pipeline** — 14 BIRD scripts, 8 Spider scripts, shared infrastructure,
   and ~35 SLURM job scripts — spanning local inference, AWS EC2, and the Northeastern
   Discovery HPC cluster (H200 GPUs).

---

## 📈 Best Result Progression

### Spider V1 dev (1,034 questions)

| Stage | Result Accuracy | Note |
|-------|----------------:|------|
| Grok-4 (frontier baseline) | 73.7% | 762/1034 |
| DeepSeek V3 (frontier baseline) | 71.8% | 742/1034 |
| **Qwen2.5-Coder-7B, SFT + DPO** | **78.2%** ✅ | **beats both frontier models** |
| (cross-eval) BIRD-DPO adapter on Spider | 75.9% | strong positive transfer (−2.3pp) |

Execution accuracy 97.3% · Row-count accuracy 86.6% · 1,040 DPO pairs · ~$25.

### BIRD dev (1,534 questions)

| # | Configuration | Model | Result Acc | Note |
|---|---------------|-------|-----------:|------|
| 1 | Base (no fine-tuning) | 7B | 27.0% | starting point |
| 2 | SFT + frontier DPO (initial, mixed pairs, cutoff 1536) | 7B | 46.9% | first published BIRD result |
| 3 | Frontier DPO, **clear_preference pairs only** (1,219) | 7B | **50.3%** ✅ | best off-policy DPO |
| 4 | **ExCoT CoT-SFT** (reasoning distillation) | 7B | **52.1%** ✅ | best single greedy 7B |
| 5 | **CoT-SFT + Best-of-N (K=8, self-consistency)** | 7B | **58.5%** 🏆 | **beats frontier; best 7B** |
| — | Grok-4.1-fast (frontier) | — | 55.4% | 849/1534 |
| — | DeepSeek V3.2 (frontier) | — | 54.7% | 839/1534 |
| 6 | Base | 14B | 54.3% | suspected pretrain contamination |
| 7 | SFT (3 epochs, cutoff 8192) | 14B | 55.0% | |
| 8 | **SFT + Best-of-N (K=4, self-consistency)** | 14B | **59.3%** 🏆 | **best overall (14B)** |

---

## 🧪 Experiment Log (by branch / phase)

The work proceeded through six branches. Each represents a distinct research phase.

### Phase 0 — `main`: The Spider DPO pipeline (complete, published)

The foundational result and the reusable infrastructure.

- Built the **6-step pipeline**: baseline eval → build preference pairs → LLM judge →
  format training → external fine-tune → evaluate.
- Ran **Grok-4** and **DeepSeek V3** on Spider dev; categorized into 4 pair types
  (clear preference, both-correct/different SQL → judge, both-wrong → gold fallback, skip).
- **Gemini 2.5 Flash** as the tie-breaking judge (position-bias controlled by seeded A/B swap),
  ~$0.04 for 478 calls.
- Produced **1,040 DPO pairs**; SFT on 7,000 Spider-train gold examples + DPO.
- **Result: 78.2%**, beating both frontier teachers. Published model + dataset to HuggingFace.
- Shared infra (`src/shared/`): unified `llm_client`, `schema_loader`, `sqlite_executor`,
  `evaluator` — reused by every later experiment.

### Phase 1 — `feat/bird-frontier-dpo`: Porting the recipe to BIRD (7B)

- Ran both frontier models on the **full BIRD train set (9,428 Q)** via OpenRouter:
  **DeepSeek V3.2 = 53.2%**, **Grok-4.1-fast = 53.5%** (both 9,428/9,428 complete).
- Rewrote `build_pairs.py` (full CLI, gold-fallback off by default), made all BIRD scripts
  **Python 3.8-compatible** for the HPC container (`from __future__ import annotations`).
- DPO training + eval scripts; cross-evaluated the BIRD adapter on Spider and BIRD mini-dev.
- **First BIRD result: 46.9%** (+20pp over base); later **validated to 50.3%** when filtered
  to clear_preference pairs only.
- **Validated finding:** judge-resolved *style* pairs (both models correct, different SQL)
  actively **hurt** BIRD (−9.6pp: 50.3% → 40.7%). BIRD's metric is result correctness, so
  style preference carries zero signal. **Rule: BIRD DPO uses clear_preference pairs only.**

### Phase 1b — Delta Learning attempt (on the same line of work)

- Implemented the "Delta Learning Hypothesis" (arXiv:2507.06187): build pairs from two *weak*
  small models (Qwen 3B @ 39.6%, Qwen 1.5B @ 27.1% on BIRD train), DPO the 7B on the deltas.
- Built **8,293 pairs** (incl. gold-fallback when both wrong).
- **Result: 35.3% (regression from 46.9%).** Diagnosed precisely: gold-fallback pairs were
  **77% of training data** and dominated the signal; DPO overfit (loss → 0.009, rewards 100%
  by epoch 1). **Clean negative result with a documented retry plan** (clean pairs only,
  β=0.05, 1 epoch).

### Phase 2 — `feat/bird-frontier-dpo-14B`: Scaling the student to 14B

- SFT + DPO scripts for **Qwen2.5-Coder-14B**; eval (base/SFT/DPO/Best-of-N/self-sampling).
- **14B base 54.3% → SFT 55.0%.**
- **Key finding — the "annotator must be stronger than the student" ceiling:** the *same*
  1,219 frontier pairs that boosted the 7B by +23pp **regressed the 14B by −5.7pp** (to 49.3%).
  Cause: the 14B base (54.3%) already matches the frontier annotators (~53%), so the "chosen"
  SQL isn't actually better → DPO becomes noise. **Off-policy frontier-DPO works for 7B,
  fails for 14B.**

### Phase 3 — `feat/bird-self-sampling-dpo`: On-policy data + Best-of-N

- **Best-of-N self-consistency at eval (no retraining):** 14B SFT + **K=4** with majority vote
  on execution result → **59.3%** (+4.3pp over SFT, **best overall result**). Critical bug fix:
  first-executable selection only gave +0.9pp; *majority voting* gave the real +3.4pp.
- **Self-sampling DPO (14B):** generated 680 (correct, wrong) pairs from the model's own
  outputs (K=4, T=0.8, 22.7% yield). Training curve healthy, but eval (SS-DPO + BoN) = **58.7%
  — slight regression.** Diagnosed as **diversity collapse**: rank_0 win rate 79.4% → 87.2%;
  worst on challenging questions (−4.1pp). **Finding: self-sampling DPO can hurt downstream
  Best-of-N by reducing sample diversity.**
- Implemented vLLM-backed inference + `eval_best_of_n.py` (GPU-generate / CPU-execute split).

### Phase 4 — Teacher bake-offs (W&B Inference)

Two rigorous "is a better teacher worth the credits?" studies, both answered with data.

- **BIRD bake-off** (250 dev Q, ~$0.65): best available teacher **Qwen3-Coder-480B ≈ 52%** —
  below the ~60% decision bar and tied with the frontier teachers already in use.
  DeepSeek-V4-Pro (1.6T generalist) only 46.8% → *"most advanced generalist" ≠ best at SQL.*
  **Decision: don't spend credits on a teacher swap.**
- **Spider bake-off** (250 dev Q, 8 models, ~$1.5): only **Qwen3-Coder-480B (81.2%)** beats the
  78.2% student (+3pp); a 550B generalist scored *below* the 7B. **Conclusion: teacher-DPO is
  closed for Spider** — pivot to execution-feedback training. Also surfaced that Spider's
  strict positional result-match depresses absolute scores (the student's true accuracy is
  likely a couple points above 78.2%).

### Phase 5 — `feat/excot-onpolicy`: Reasoning distillation (the 7B breakthrough)

ExCoT = iterative DPO with chain-of-thought, execution-feedback only (arXiv:2503.19988).

- **CoT seed generation:** Qwen3-Coder-480B writes CoT for BIRD train, execution-filtered to
  keep-correct → **5,593 seed examples** from 9,428 (59% yield, ~$8 W&B credits, CPU+API only).
- **CoT-SFT** on the *base* 7B (2 epochs, cutoff 8192) → **52.1%** — beats the frontier-DPO
  prior (50.3%) and the self-sampling prior. **Validated finding: distilling the teacher's
  *reasoning* generalizes across BIRD's cross-domain dev DBs better than distilling its *SQL
  answers*.**
- **CoT-SFT + Best-of-N (K=8, self-consistency) → 58.5%** on the full dev set — **new 7B best,
  beats both frontier models.**
- **On-policy CoT-DPO (Stage 2.3): FAILED** (−8.2pp). 514 CoT pairs too few; long similar CoT
  chains give weak contrast + length bias; rewards stuck at chance. **Rule: for 7B ExCoT,
  CoT-SFT is the win; vanilla on-policy CoT-DPO hurts — Best-of-N is the higher-ROI lever.**

### Phase 6 — `feat/bird-7b-clean`: GRPO (execution-reward RL) — in progress

- **GRPO** via TRL `GRPOTrainer` with a **binary SQL-execution reward** (run generated SQL,
  reward = result-set match vs gold). `grpo_train.py` + `grpo_dataset.py`.
- **Fixed the broken TRL API** (June 2026): `kl_coef→beta`, `max_new_tokens→max_completion_length`,
  `tokenizer→processing_class`, added `max_prompt_length=3072`, enforced `batch_size==num_gen`
  (GRPO divisibility); clean env `grpo_env` (torch 2.5.1, trl 0.15.2) keeping vLLM out.
- **Validated working:** reward climbs **0.46 → 0.65** on train; ~2.5 min/step on H200.
- Three runnable configs: `bird_grpo_7b_{quick,from_sft,from_dpo}.sh` (quick = 600-Q subset
  sized to fit a 4h SLURM wall).
- **Key HPC fix:** `/home` is not mounted on GPU compute nodes — staged all adapters/data to
  `/scratch` (a silent fall-back to a fresh LoRA had been caused by an invisible `/home` path).

---

## 🔬 Validated Research Findings

These are reusable, evidence-backed conclusions — the intellectual core of the project.

1. **A specialized 7B can beat 100B+ frontier models** on a narrow task (Spider: 78.2% vs ~73%).
2. **Frontier disagreement is viable DPO signal** — correctness pairs + judged style pairs work
   on Spider.
3. **The DPO annotator must be meaningfully stronger than the student.** Same pairs: +23pp on a
   7B, −5.7pp on a 14B whose base already matched the annotators.
4. **For execution-metric benchmarks (BIRD), use correctness pairs only.** Style-preference pairs
   carry no signal and regress accuracy.
5. **Reasoning distillation > answer distillation cross-domain.** CoT-SFT beat SQL-answer DPO on
   BIRD's unseen dev databases.
6. **Best-of-N with execution-based *majority voting* is the highest-ROI lever** (7B 52.1%→58.5%;
   14B 55.0%→59.3%) — inference-time only, no retraining. *First-executable* selection barely
   moves the needle; the majority vote is what matters.
7. **Self-sampling DPO can collapse output diversity** and *hurt* downstream Best-of-N, worst on
   the hard questions that need exploration most.
8. **Teacher quality was never the lever once the student is near-frontier** (both bake-offs).
9. **Scale ≠ SQL ability:** coding-specialist 480B beats 550B–1.6T generalists at text-to-SQL.
10. **Vanilla on-policy CoT-DPO is unstable on a 7B** (length bias, weak contrast on long chains).

---

## 🛠️ Engineering & Infrastructure Achievements

- **Benchmark-agnostic shared core** (`src/shared/`): one LLM client (xAI + OpenRouter +
  self-contained W&B clients), schema loader, SQLite executor, and result-set evaluator reused
  by every experiment so training and eval metrics always match.
- **Full toolchain per method:** pipelines, pair builders, judges, formatters, SFT/DPO/GRPO
  trainers, single-greedy and Best-of-N evaluators — for both 7B and 14B, direct-SQL and CoT.
- **Multi-platform execution:** local (transformers + peft + bitsandbytes 4-bit QLoRA),
  AWS EC2 g5.xlarge (A10G 24GB), and Northeastern Discovery HPC (H200 80GB, Apptainer, SLURM).
- **Solved hard environment problems, documented for reuse:**
  - Python 3.8 HPC container compat across all scripts.
  - Dependency triple that holds the whole pipeline together:
    `transformers 4.46.3 + trl 0.12.2 + vllm 0.6.3`.
  - GPU-generate / CPU-execute **split into separate SLURM jobs** to dodge the idle-GPU watchdog
    during long SQL-execution phases.
  - `/home`-not-mounted-on-compute-nodes gotcha → stage everything to `/scratch`.
  - pip wheel pitfalls on py3.8 (`openai==1.40.0`, `httpx<0.28`).
- **Resumable, checkpointed evals** (every 25–50 Q) to survive SLURM wall-clock limits.

---

## 📦 Published & Stored Artifacts

- **HuggingFace model:** `jk200201/qwen2.5-coder-7b-sql-dpo` (Spider 78.2% adapter)
- **HuggingFace dataset:** `jk200201/spider-dpo-1040` (1,040 DPO pairs + SFT data)
- **Local adapters:** `models/qwen-7b-sql-dpo/` (Spider), `models/bird_dpo/`, `bird_sft_adapter_1/`
- **HPC checkpoints (on `/scratch/phalle.y` unless noted):**
  - `bird_sft_adapter_7b/final_adapter` — 7B SFT base for GRPO
  - `bird_frontier_dpo_adapter/final_adapter` — 50.3% best off-policy DPO
  - `bird_cot_sft_adapter_7b/final_adapter` — **52.1% best single greedy 7B**
  - 14B SFT adapter — 55.0% (Best-of-N base for the 59.3% result)
- **Docs:** `README.md` (Spider), `BIRD_RESULTS.md` (BIRD gap analysis), `CLAUDE.md` (full log).

---

## 🌿 Branch Map

| Branch | Phase | What it holds |
|--------|-------|---------------|
| `main` | 0 | Spider 78.2% pipeline + shared infra + BIRD frontier pipeline + GRPO infra |
| `feat/bird-frontier-dpo` | 1 | BIRD frontier-DPO (7B), full pair-building rewrite, cross-eval |
| `feat/bird-frontier-dpo-14B` | 2 | 14B SFT/DPO/eval; annotator-ceiling finding |
| `feat/bird-self-sampling-dpo` | 3 | Self-sampling DPO + Best-of-N self-consistency (59.3% best) |
| `feat/excot-onpolicy` | 5 | CoT seed gen, CoT-SFT (52.1%), CoT-DPO (failed), CoT Best-of-N (58.5% best 7B) |
| `feat/bird-7b-clean` | 6 | GRPO execution-reward RL (working, in progress) — **current branch** |

---

## 💰 Cost Summary

| Item | Cost |
|------|------|
| Spider full pipeline (Grok + DeepSeek + judge + EC2) | ~$25–30 |
| BIRD frontier pipeline (Grok + DeepSeek on 9,428 Q + judge + EC2) | ~$30–35 |
| CoT seed generation (Qwen3-Coder-480B via W&B) | ~$8 |
| Teacher bake-offs (BIRD ~$0.65 + Spider ~$1.5) | ~$2 |
| HPC compute (Discovery H200, SLURM) | academic allocation |
| **Approx. cash spend across the whole project** | **~$70–75** |

---

*Compiled 2026-06-21 from git history across 6 branches, project memory, `README.md`,
`BIRD_RESULTS.md`, and `CLAUDE.md`. Some BIRD numbers were re-validated after the first run
(e.g. 46.9% → 50.3% with cleaner pairs); both are shown so the progression is honest.*
