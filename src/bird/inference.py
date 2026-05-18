"""
Shared inference utilities for BIRD pipelines.

Provides:
- build_instruction(): the canonical prompt format (must match training)
- BIRDvLLMEngine: vLLM-backed batched generation with optional Best-of-N
- _flash_attn_available(): helper used by trainer scripts

This module is imported by:
  - src.bird.eval_finetuned         (build_instruction)
  - src.bird.build_sft_data         (build_instruction)
  - src.bird.build_pairs            (build_instruction)
  - src.bird.sft_train              (_flash_attn_available)
  - src.bird.dpo_train              (_flash_attn_available)
  - src.bird.eval_best_of_n         (BIRDvLLMEngine)
  - src.bird.build_self_sampling_pairs (BIRDvLLMEngine)
"""
from __future__ import annotations

import re
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — must match exactly what was used during DPO training
# Single source of truth across the codebase
# ─────────────────────────────────────────────────────────────────────────────

def build_instruction(question: str, schema: str, evidence: str = "") -> str:
    """Canonical BIRD prompt format. Used by training data builders + eval."""
    evidence_block = f"External Knowledge:\n{evidence}\n\n" if evidence.strip() else ""
    return (
        "Convert the following natural language question into a valid SQL query.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{evidence_block}"
        f"Question: {question}\n\n"
        "Return only the SQL query with no explanation."
    )


SYSTEM_PROMPT = (
    "You are an expert SQL query generator. "
    "Given a database schema and a natural language question, "
    "generate a single valid SQL query that answers the question. "
    "Output ONLY the SQL query, nothing else."
)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT POSTPROCESSING — strip markdown fences, add trailing semicolon
# ─────────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    """Strip markdown fences, return SQL with trailing semicolon."""
    text  = text.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    return text if text.endswith(";") else text + ";"


# ─────────────────────────────────────────────────────────────────────────────
# FLASH-ATTENTION HELPER — used by SFT + DPO trainers
# ─────────────────────────────────────────────────────────────────────────────

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# vLLM BATCHED INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BIRDvLLMEngine:
    """
    vLLM-backed batched SQL generation. Supports K-sample generation for
    Best-of-N inference and self-sampling preference pair generation.

    Loads base model + (optional) LoRA adapter once. All subsequent
    generate() calls reuse the loaded engine.

    Usage:
        engine = BIRDvLLMEngine(
            base_model="Qwen/Qwen2.5-Coder-14B-Instruct",
            adapter_path="/scratch/.../bird_sft_adapter_14b/checkpoint-3100",
        )
        # Each item is a tuple (question, schema, evidence)
        prompts = [(q, s, e) for q, s, e in items]
        # Returns: list of (list of K candidate SQL strings)
        candidates_per_question = engine.generate(prompts, k=4, temperature=0.8)
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=base_model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,  # safe upper bound; covers r=32 with margin
            trust_remote_code=True,
        )

        self.lora_request = None
        if adapter_path:
            self.lora_request = LoRARequest(
                lora_name="bird_adapter",
                lora_int_id=1,
                lora_path=adapter_path,
            )
            print(f"vLLM engine loaded with adapter: {adapter_path}")
        else:
            print(f"vLLM engine loaded (base model only, no adapter)")

    def _build_chat_prompt(self, question: str, schema: str, evidence: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_instruction(question, schema, evidence)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        items: Iterable[tuple[str, str, str]],
        k: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> list[list[str]]:
        """
        Args:
            items: iterable of (question, schema, evidence) tuples
            k: number of candidates per question (1 = greedy, K>1 = sampling)
            temperature: 0.0 for greedy, 0.7-1.0 for sampling
            max_tokens: max generated tokens per candidate

        Returns:
            list of length len(items); each element is a list of K SQL strings
        """
        from vllm import SamplingParams

        items = list(items)
        prompts = [self._build_chat_prompt(q, s, e) for q, s, e in items]

        sampling_params = SamplingParams(
            n=k,
            temperature=temperature if k > 1 else 0.0,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
        )

        results: list[list[str]] = []
        for out in outputs:
            cands = [extract_sql(o.text) for o in out.outputs]
            results.append(cands)
        return results
