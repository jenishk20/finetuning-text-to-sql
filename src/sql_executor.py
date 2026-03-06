"""
SQL executor for running queries against the local MySQL GameDB.

Connects to the database, executes a SQL string, and returns
either the result set or the error message.
"""

import os
import time

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Create a new MySQL connection from environment variables."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "retrogamedb_2026"),
    )


def execute_query(sql: str, timeout: int = 30) -> dict:
    """
    Execute a SQL query against the configured MySQL database.

    Handles multi-statement SQL (e.g., DROP TABLE + CREATE TABLE for Q5)
    by splitting on semicolons and executing each statement.

    Returns a dict with:
      - "success": bool
      - "columns": list of column names (if success and SELECT)
      - "rows": list of tuples (if success and SELECT)
      - "row_count": number of rows returned
      - "execution_time_ms": time taken in milliseconds
      - "error": error message string (if failure)
    """
    conn = None
    try:
        conn = get_connection()
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
            else:
                # DDL or DML statement — commit it
                conn.commit()

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
        if conn and conn.is_connected():
            conn.close()
