"""
Flexible LLM client supporting multiple providers (xAI, OpenRouter).

All providers use the OpenAI-compatible API format, so we just
swap the base_url and api_key.
"""

from __future__ import annotations

import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "default_model": "grok-4.1-fast",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "deepseek/deepseek-v3.2",
    },
}

# Popular models on OpenRouter (for reference / listing)
OPENROUTER_MODELS = {
    "gpt-4o":        "openai/gpt-4o",
    "gpt-4.1":       "openai/gpt-4.1",
    "gpt-4.1-mini":  "openai/gpt-4.1-mini",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "deepseek-v3":   "deepseek/deepseek-v3.2",          # DeepSeek V3.2 — $0.26/$0.38 per M
    "deepseek-r1":   "deepseek/deepseek-r1",
    "llama-70b":     "meta-llama/llama-3.1-70b-instruct",
    "llama-8b":      "meta-llama/llama-3.1-8b-instruct",
    "gemini-flash":  "google/gemini-2.0-flash-001",
    "gemini-pro":    "google/gemini-2.5-pro-preview-03-25",
    "qwen-72b":      "qwen/qwen-2.5-coder-32b-instruct",
    "mistral-large": "mistralai/mistral-large",
}

SQLITE_SYSTEM_PROMPT = """You are an expert SQLite SQL query generator.
You are given a database schema (CREATE TABLE statements) and a natural language question.
Your job is to generate a single valid SQLite SELECT query that answers the question.

Rules:
- Output ONLY the SQL query. No explanations, no markdown, no code fences.
- Use SQLite syntax. Do NOT use MySQL-specific features like backticks, GROUP_CONCAT with SEPARATOR, or IFNULL.
- SQLite uses double quotes for identifiers and single quotes for strings.
- If the question asks for all rows from a table to appear, use LEFT JOIN (not INNER JOIN).
- Do NOT use LIMIT unless explicitly asked.
- End the query with a semicolon.
"""


def list_available_models():
    """Print all shortcut model names available via OpenRouter."""
    print("\n  Available model shortcuts (for --model flag):")
    print(f"  {'Shortcut':<18} {'Full Model ID'}")
    print(f"  {'-' * 55}")
    for shortcut, full_id in OPENROUTER_MODELS.items():
        print(f"  {shortcut:<18} {full_id}")
    print(f"\n  You can also pass any full model ID directly (e.g., openai/gpt-4o)")
    print(f"  For xAI, use --provider xai (default model: grok-4-1-fast-reasoning)\n")


def resolve_model(model_name: str, provider: str) -> str:
    """Resolve a shortcut name to a full model ID."""
    if provider == "xai":
        return model_name

    # Check if it's a shortcut
    if model_name in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[model_name]

    # Otherwise assume it's a full model ID
    return model_name


def get_client(provider: str) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise ValueError(
            f"Missing API key. Set {config['api_key_env']} in your .env file."
        )

    return OpenAI(api_key=api_key, base_url=config["base_url"])


def extract_sql(response_text: str) -> str:
    """Strip markdown fences, thinking tags, or extra text from model response."""
    text = response_text.strip()

    # Remove <think>...</think> blocks (DeepSeek R1 and similar reasoning models)
    think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    return text


def call_llm(
    question: str,
    schema: str,
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    evidence: str | None = None,
) -> dict:
    """
    Call any LLM with schema + question, return generated SQL.

    Returns dict with:
      - "generated_sql": the SQL string
      - "model": full model name used
      - "provider": provider name
      - "usage": token usage dict
      - "raw_response": full response text from API
    """
    client = get_client(provider)
    full_model = resolve_model(model, provider)

    messages = [
        {"role": "system", "content": SQLITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"### Database Schema:\n{schema}\n\n"
                + (f"### External Knowledge:\n{evidence}\n\n" if evidence else "")
                + f"### Question:\n{question}\n\n"
                f"### SQLite Query:"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=full_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_text = response.choices[0].message.content or ""
    sql = extract_sql(raw_text)

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "generated_sql": sql,
        "model": full_model,
        "provider": provider,
        "usage": usage,
        "raw_response": raw_text,
    }
