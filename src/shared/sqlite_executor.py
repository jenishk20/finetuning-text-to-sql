"""
SQL executor for running queries against Spider's SQLite databases.
"""

import sqlite3
import time


def execute_sqlite_query(sql: str, db_path: str, timeout: int = 30) -> dict:
    """
    Execute a SQL query against a SQLite database file.

    Returns a dict with:
      - "success": bool
      - "columns": list of column names
      - "rows": list of tuples
      - "row_count": number of rows
      - "execution_time_ms": time taken
      - "error": error message (if failure)
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.text_factory = str
        cursor = conn.cursor()

        start = time.time()

        statements = [s.strip() for s in sql.split(";") if s.strip()]

        columns = []
        rows = []

        for stmt in statements:
            cursor.execute(stmt)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

        elapsed_ms = round((time.time() - start) * 1000, 2)

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": elapsed_ms,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0,
            "error": str(e),
        }

    finally:
        if conn:
            conn.close()
