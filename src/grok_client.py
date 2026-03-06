"""
Grok API client for Text-to-SQL generation.

Uses the xAI API (OpenAI-compatible) to send database schema + natural language
question and receive a generated MySQL query.
"""

import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MYSQL_SYSTEM_PROMPT = """You are an expert MySQL SQL query generator.
You are given a database schema (CREATE TABLE statements) and a natural language question.
Your job is to generate a single valid MySQL SELECT query that answers the question.

Rules:
- Output ONLY the SQL query. No explanations, no markdown, no code fences.
- Use MySQL syntax (e.g., GROUP_CONCAT, backticks for reserved words like `user`).
- If the question asks for all rows from a table to appear, use LEFT JOIN (not INNER JOIN).
- If the question asks to create a table, output the full DDL statements needed.
- Do NOT use LIMIT unless explicitly asked.
- End the query with a semicolon.
"""

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


def build_prompt(schema: str, question: str, dialect: str = "mysql") -> list[dict]:
    """Build the chat messages for the Grok API call."""
    system_prompt = SQLITE_SYSTEM_PROMPT if dialect == "sqlite" else MYSQL_SYSTEM_PROMPT
    dialect_label = "SQLite" if dialect == "sqlite" else "MySQL"

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"### Database Schema:\n{schema}\n\n"
                f"### Question:\n{question}\n\n"
                f"### {dialect_label} Query:"
            ),
        },
    ]


def extract_sql(response_text: str) -> str:
    """Strip markdown fences or extra text from the model response."""
    text = response_text.strip()

    # Remove ```sql ... ``` wrapping if present
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    return text


def call_grok(question: str, schema: str, config: dict, dialect: str = "mysql") -> dict:
    """
    Call the xAI API with schema + question, return generated SQL.

    Returns dict with:
      - "generated_sql": the SQL string
      - "model": model name used
      - "usage": token usage dict
      - "raw_response": full response text from API
    """
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )

    messages = build_prompt(schema, question, dialect=dialect)

    response = client.chat.completions.create(
        model=config["model"]["name"],
        messages=messages,
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
    )

    raw_text = response.choices[0].message.content
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
        "model": config["model"]["name"],
        "usage": usage,
        "raw_response": raw_text,
    }
