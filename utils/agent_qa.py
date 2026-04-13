from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .client import SnowflakeAgentClient
from .config import LLM_POPULATION_GUIDANCE, MAX_PREVIEW_ROWS
from .models import FinalAnswer, QueryExecutionResult

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_SYSTEM = (
    """\
You are a data analyst assistant for a US population rent burden chatbot.

Answer the user's question using ONLY the data rows returned from the database query below.
Rules:
- Do NOT invent, estimate, or reference any numbers not present in the results.
- Use specific numbers from the data (percentages, counts, state/county names).
- If the result has multiple rows, summarize the key findings concisely.
- Be friendly, clear, and informative. Aim for 2-5 sentences.
- If the result set is empty, say no data was found rather than guessing.
- If the conversation history shows this is a follow-up question, acknowledge
  the context naturally (e.g. "Compared to California you asked about earlier…").

"""
    + LLM_POPULATION_GUIDANCE
)

_USER_TMPL = """\
{history_block}Question: {question}

Source table(s): {tables}
SQL executed: {sql}
Rows returned: {row_count}

Data (up to {preview_rows} rows shown):
{rows_formatted}
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


def _format_rows(rows: List[Dict[str, Any]], preview: int) -> str:
    if not rows:
        return "(no rows)"
    lines = []
    for i, row in enumerate(rows[:preview], 1):
        parts = ", ".join(f"{k}={v}" for k, v in row.items())
        lines.append(f"  Row {i}: {parts}")
    if len(rows) > preview:
        lines.append(f"  ... and {len(rows) - preview} more rows not shown")
    return "\n".join(lines)


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def call_qa_agent(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    source_tables: List[str],
    sql_text: str,
    query_result: QueryExecutionResult,
    history: Optional[list] = None,
) -> FinalAnswer:
    """
    Use the Cortex LLM to produce a grounded natural-language answer.

    The LLM is shown only the returned rows and must answer strictly from them.
    Returns FinalAnswer with ok=True and the full result data attached.
    """
    history_block = _build_history_block(history or [])
    rows_formatted = _format_rows(query_result.rows, MAX_PREVIEW_ROWS)
    user_prompt = _USER_TMPL.format(
        history_block=history_block,
        question=user_question,
        tables=", ".join(source_tables),
        sql=sql_text,
        row_count=query_result.row_count,
        preview_rows=MAX_PREVIEW_ROWS,
        rows_formatted=rows_formatted,
    )
    user_message = client.call_cortex(_SYSTEM, user_prompt)

    return FinalAnswer(
        ok=True,
        status="SUCCESS",
        user_message=user_message.strip(),
        route="qa_agent",
        source_tables=source_tables,
        sql=sql_text,
        row_count=query_result.row_count,
        data=query_result.rows,
        debug={"columns": query_result.columns},
    )
