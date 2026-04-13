from __future__ import annotations

import json
from typing import List, Optional

from .client import SnowflakeAgentClient, parse_llm_json
from .config import LLM_POPULATION_GUIDANCE, MAX_RESULT_ROWS, TABLE_CATALOG
from .models import LocationContext, SqlGenerationResult

# ----------------------------------------------------------
# Schema builder — injects only the selected table(s) into the prompt
# ----------------------------------------------------------

def _build_schema_text(tables: List[str]) -> str:
    """
    Build a rich schema block for the selected tables, including:
      - Table description and grain
      - Per-column: SQL data type + description
      - Important notes (quoting requirements, ratio interpretation)
      - Sample values for key filter columns
    """
    blocks = []
    for table_name in tables:
        info = TABLE_CATALOG.get(table_name)
        if not info:
            continue

        lines = []
        lines.append(f"━━━ Table: {table_name}  (grain: {info['grain']}) ━━━")
        lines.append(f"Description: {info['description']}")
        lines.append("")

        # Column schema with types
        lines.append("Columns (name | SQL type | description):")
        for col_name, col_val in info["columns"].items():
            if isinstance(col_val, tuple):
                dtype, desc = col_val
            else:
                dtype, desc = "VARCHAR", col_val
            lines.append(f"  {col_name:<58} {dtype:<10}  {desc}")

        # Notes (quoting rules, semantics)
        if info.get("notes"):
            lines.append("")
            lines.append(f"IMPORTANT NOTES: {info['notes']}")

        # Sample values
        if info.get("sample_values"):
            lines.append("")
            lines.append("Sample values for key columns (use these to construct filters):")
            for col, sample in info["sample_values"].items():
                lines.append(f"  {col}: {sample}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ----------------------------------------------------------
# Prompt
# ----------------------------------------------------------

_SYSTEM_TMPL = """\
You are a Snowflake SQL generator for a US rent burden chatbot.

Schema of queryable tables:
{schema}

Strict rules — violations will cause the query to be rejected:
1. Only SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, CREATE,
   ALTER, TRUNCATE, MERGE, GRANT, REVOKE, CALL, EXECUTE.
2. Always end the query with LIMIT {limit}.
3. Use only columns that appear in the schema above. Do not invent column names.
4. For string filters use ILIKE for case-insensitive matching, e.g.
   WHERE STATE_NAME ILIKE '%california%'
5. Column names that contain spaces or special characters must be wrapped in
   double quotes, e.g. "Total: Renter-occupied housing units"
6. Do not JOIN to tables not listed in the schema.
7. Ratio columns (rent_burden_ratio_30_plus, etc.) are decimal fractions
   between 0 and 1, not percentages. Multiply by 100 in SELECT if the user
   wants a percentage.

{population_guidance}

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "ok": true | false,
  "sql": "SELECT ... LIMIT {limit}",
  "message": "explanation only when ok is false, empty string otherwise"
}}
"""

_USER_TMPL = """\
{history_block}Question: {question}
Tables to use: {tables}
"""

_USER_RETRY_TMPL = """\
{history_block}Question: {question}
Tables to use: {tables}

IMPORTANT — your previous SQL attempt failed with this error:
  {previous_error}

Previous failing SQL:
  {previous_sql}

Generate a corrected SQL query that avoids the error above.
"""

_HISTORY_TMPL = """\
[RECENT CONVERSATION — last {n} turn(s)]
{turns}

"""


def _build_history_block(history: list) -> str:
    if not history:
        return ""
    lines = []
    for i, turn in enumerate(history, 1):
        lines.append(f"Turn {i}: User: {turn['user']}")
        lines.append(f"        Bot : {turn['bot'][:120]}{'…' if len(turn['bot']) > 120 else ''}")
    return _HISTORY_TMPL.format(n=len(history), turns="\n".join(lines))


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def call_generate_sql(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    tables: List[str],
    grain: Optional[str],
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
    history: Optional[list] = None,
    location_ctx: Optional[LocationContext] = None,
) -> SqlGenerationResult:
    """
    Use the Cortex LLM to generate a safe, constrained Snowflake SELECT query.

    On retry attempts, pass previous_sql and previous_error so the LLM
    knows what went wrong and can produce a corrected query.
    """
    schema_text = _build_schema_text(tables)
    system_prompt = _SYSTEM_TMPL.format(
        schema=schema_text,
        limit=MAX_RESULT_ROWS,
        population_guidance=LLM_POPULATION_GUIDANCE,
    )

    history_block = _build_history_block(history or [])
    loc_block = location_ctx.as_prompt_block() if location_ctx and location_ctx.is_set() else ""

    if previous_sql and previous_error:
        user_prompt = _USER_RETRY_TMPL.format(
            history_block=history_block,
            question=user_question,
            tables=", ".join(tables),
            previous_error=previous_error,
            previous_sql=previous_sql,
        )
    else:
        user_prompt = _USER_TMPL.format(
            history_block=history_block,
            question=user_question,
            tables=", ".join(tables),
        )
    # Prepend location block after history so the LLM has canonical names
    # available for WHERE clause generation before it reads the question.
    if loc_block:
        user_prompt = loc_block + user_prompt

    raw = client.call_cortex(system_prompt, user_prompt)

    try:
        data = parse_llm_json(raw)
        return SqlGenerationResult(
            ok=bool(data.get("ok", False)),
            sql=data.get("sql"),
            message=data.get("message", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return SqlGenerationResult(
            ok=False,
            sql=None,
            message=f"SQL generator returned unparseable response: {raw[:200]}",
        )
