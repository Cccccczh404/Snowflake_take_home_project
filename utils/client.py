from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict

import pandas as pd
from snowflake.snowpark import Session

from .config import CORTEX_MODEL, MAX_RESULT_ROWS, LOCAL_CSV_FILES
from .fuzzy_location import FuzzyLocationResolver
from .models import QueryExecutionResult


class SnowflakeAgentClient:
    """
    Two separate backends:
      - Snowflake Snowpark Session  → Cortex LLM calls only (call_cortex)
      - SQLite in-memory database   → all data queries (query_rows)

    The local CSVs in ./output/ are loaded into SQLite at startup so no
    Snowflake tables are required. Snowflake is only contacted for LLM inference.
    """

    def __init__(self, connection_kwargs: Dict[str, Any]) -> None:
        # Snowflake — LLM only
        self.session = Session.builder.configs(connection_kwargs).create()
        # Local SQLite — data queries
        self._db = self._load_local_db()
        # Offline fuzzy location resolver — built from actual DB rows
        self._location_resolver = self._build_location_resolver()

    # ----------------------------------------------------------
    # Local DB bootstrap
    # ----------------------------------------------------------

    @staticmethod
    def _load_local_db() -> sqlite3.Connection:
        """
        Read every CSV from ./output/ into a SQLite in-memory database.
        Each file becomes a table whose name matches the TABLE_CATALOG keys.
        """
        con = sqlite3.connect(":memory:")
        # Allow double-quoted identifiers (needed for CBG columns with colons/spaces)
        con.execute("PRAGMA case_sensitive_like = OFF")
        for table_name, csv_path in LOCAL_CSV_FILES.items():
            print(f"  [LOCAL DB] loading {csv_path} → {table_name} ...", flush=True)
            df = pd.read_csv(csv_path, dtype_backend="numpy_nullable")
            df.to_sql(table_name, con, if_exists="replace", index=False)
            print(f"  [LOCAL DB] {table_name}: {len(df):,} rows, {len(df.columns)} columns", flush=True)
        return con

    def _build_location_resolver(self) -> FuzzyLocationResolver:
        """Build offline fuzzy index from the actual state/county names in the DB."""
        states = [
            r[0] for r in self._db.execute(
                "SELECT DISTINCT STATE_NAME FROM STATE_SUMMARY WHERE STATE_NAME IS NOT NULL"
            ).fetchall()
        ]
        counties = [
            r[0] for r in self._db.execute(
                "SELECT DISTINCT COUNTY_NAME FROM COUNTY_SUMMARY WHERE COUNTY_NAME IS NOT NULL"
            ).fetchall()
        ]
        print(
            f"  [FUZZY] index built: {len(states)} states, {len(counties)} counties",
            flush=True,
        )
        return FuzzyLocationResolver(states, counties)

    def augment_question(
        self, question: str
    ) -> tuple:
        """
        Resolve fuzzy location mentions offline.

        Returns (augmented_question, clarifications) where:
          augmented_question — question with HIGH-confidence names injected
          clarifications     — list of (original_token, canonical, loc_type)
                               for MEDIUM-confidence matches needing user confirmation
        """
        return self._location_resolver.augment_question(question)

    def close(self) -> None:
        self.session.close()
        self._db.close()

    # ----------------------------------------------------------
    # LLM  (Snowflake Cortex)
    # ----------------------------------------------------------

    def call_cortex(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call SNOWFLAKE.CORTEX.COMPLETE with a combined plain-text prompt.
        Uses $$ dollar-quoting so multiline text is passed safely without
        Snowflake's ESCAPE_STRING_SEQUENCES mangling \\n sequences.
        """
        combined = (
            f"[INSTRUCTIONS]\n{system_prompt}"
            f"\n\n[USER REQUEST]\n{user_prompt}"
            f"\n\n[YOUR RESPONSE]\n"
        )
        combined_sql = combined.replace("$$", "$ $")
        sql = (
            f"SELECT SNOWFLAKE.CORTEX.COMPLETE("
            f"'{CORTEX_MODEL}', $${combined_sql}$$) AS response"
        )
        rows = self.session.sql(sql).collect()
        raw = rows[0]["RESPONSE"]
        return str(raw).strip() if raw is not None else ""

    # ----------------------------------------------------------
    # Data query  (local SQLite)
    # ----------------------------------------------------------

    def query_rows(self, sql: str) -> QueryExecutionResult:
        """
        Execute a SELECT statement against the local SQLite database.

        SQLite adaptations applied automatically:
          - ILIKE  → LIKE  (SQLite LIKE is already case-insensitive for ASCII)
        """
        adapted = _adapt_sql_for_sqlite(sql)
        try:
            cursor = self._db.execute(adapted)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows_raw = cursor.fetchmany(MAX_RESULT_ROWS)
            rows = [dict(zip(columns, r)) for r in rows_raw]
            return QueryExecutionResult(
                ok=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                message="Query executed successfully.",
            )
        except Exception as exc:
            return QueryExecutionResult(
                ok=False,
                message=f"Query execution failed: {exc}",
            )


def _adapt_sql_for_sqlite(sql: str) -> str:
    """Convert Snowflake-specific SQL syntax to SQLite equivalents."""
    # ILIKE is case-insensitive LIKE — SQLite LIKE already is for ASCII
    sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)
    return sql


# ----------------------------------------------------------
# Shared JSON parsing helper
# ----------------------------------------------------------

def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Strip optional markdown code fences and parse JSON from LLM output.
    Raises json.JSONDecodeError if the text is not valid JSON after stripping.
    """
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    return json.loads(text.strip())
