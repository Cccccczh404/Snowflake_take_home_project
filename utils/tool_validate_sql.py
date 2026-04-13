from __future__ import annotations

import json
from typing import List

from .client import SnowflakeAgentClient, parse_llm_json
from .config import (
    FORBIDDEN_SQL_KEYWORDS,
    LLM_POPULATION_GUIDANCE,
    MAX_RESULT_ROWS,
    TABLE_CATALOG,
)
from .models import SqlValidationResult

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_ALLOWED_TABLES_TEXT = ", ".join(TABLE_CATALOG.keys())
_FORBIDDEN_KW_TEXT = ", ".join(FORBIDDEN_SQL_KEYWORDS)

_SYSTEM_TMPL = """\
You are a Snowflake SQL safety validator for a US rent burden chatbot.

Your job is to inspect the SQL query below and decide whether it is safe
and correct. If it has minor fixable issues (wrong LIMIT, missing LIMIT,
capitalisation of table/column names), rewrite it rather than rejecting it.
Only set "ok" to false when the SQL is fundamentally unsafe or unanswerable.

Validation rules to enforce:
1. Must be a SELECT statement — no INSERT, UPDATE, DELETE, DROP, CREATE,
   ALTER, TRUNCATE, MERGE, GRANT, REVOKE, CALL, EXECUTE or any DDL/DML.
   Forbidden keywords: {forbidden_kw}
2. Must reference ONLY the allowed tables for this query: {allowed_tables}
   (Other tables in the database exist but must NOT be queried here.)
3. Must end with LIMIT {limit}. If the LIMIT is missing or exceeds {limit},
   add or cap it in your rewritten version.
4. Column names with spaces or special characters must be double-quoted.
5. String comparisons must use ILIKE (Snowflake is case-insensitive by default).
6. The query must be syntactically valid Snowflake SQL.

The user's original question (for context only — do NOT change the query's
intent, only fix safety/syntax issues):
  {user_question}

{population_guidance}

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "ok": true | false,
  "rewritten_sql": "corrected SQL string, or null if ok is false and not fixable",
  "issues_found": ["list of issues you detected, empty list if none"],
  "message": "explanation only when ok is false, empty string otherwise"
}}
"""

_USER_TMPL = """\
Allowed tables for this query: {allowed_tables}
SQL to validate:
{sql}
"""


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def call_validate_sql(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    sql_text: str,
    tables: List[str],
) -> SqlValidationResult:
    """
    LLM-based SQL validation via Cortex.

    The model inspects the generated SQL against safety rules and the allowed
    table list. Minor issues (missing LIMIT, capitalisation) are auto-corrected
    in rewritten_sql. Fundamental violations (DML, wrong tables) return ok=False.
    """
    allowed_tables_str = ", ".join(tables)
    system_prompt = _SYSTEM_TMPL.format(
        forbidden_kw=_FORBIDDEN_KW_TEXT,
        allowed_tables=allowed_tables_str,
        limit=MAX_RESULT_ROWS,
        user_question=user_question,
        population_guidance=LLM_POPULATION_GUIDANCE,
    )
    user_prompt = _USER_TMPL.format(
        allowed_tables=allowed_tables_str,
        sql=sql_text,
    )
    raw = client.call_cortex(system_prompt, user_prompt)

    try:
        data = parse_llm_json(raw)
        ok = bool(data.get("ok", False))
        rewritten = data.get("rewritten_sql") or None
        message = data.get("message", "")
        issues = data.get("issues_found", [])

        # Even when ok=True, prefer the rewritten SQL if the model produced one
        # (it may have added/capped LIMIT or fixed quoting).
        if ok and rewritten and rewritten.strip() == sql_text.strip():
            rewritten = None  # no actual change — don't flag as rewritten

        return SqlValidationResult(
            ok=ok,
            message=message or ("; ".join(issues) if issues else "SQL passed validation."),
            rewritten_sql=rewritten if ok else None,
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        # If LLM returns unparseable output, fail safe and reject the SQL
        return SqlValidationResult(
            ok=False,
            message=f"SQL validator returned unparseable response: {raw[:200]}",
        )
