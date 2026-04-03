"""
Schema loader for Spider databases.

Extracts CREATE TABLE DDL from SQLite database files for use in prompts.
"""

import sqlite3
from pathlib import Path


def get_schema_from_sqlite(db_path: str) -> str:
    """
    Extract CREATE TABLE statements directly from a SQLite database file.
    This is the most reliable way since it matches the actual DB structure.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    )
    tables = cursor.fetchall()
    conn.close()

    ddl_statements = []
    for (sql,) in tables:
        if sql and "sqlite_sequence" not in sql.lower():
            ddl_statements.append(sql.strip() + ";")

    return "\n\n".join(ddl_statements)


def get_db_path(spider_data_dir: str, db_id: str) -> str:
    """Resolve the full path to a Spider SQLite database file."""
    base = Path(spider_data_dir) / "database" / db_id / f"{db_id}.sqlite"
    if base.exists():
        return str(base)

    # Fallback: search for any .sqlite file in the directory
    db_dir = Path(spider_data_dir) / "database" / db_id
    if db_dir.exists():
        for f in db_dir.glob("*.sqlite"):
            return str(f)

    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {spider_data_dir}")
