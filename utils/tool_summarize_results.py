from __future__ import annotations

from typing import Any, Dict, List

from .client import SnowflakeAgentClient
from .config import LLM_POPULATION_GUIDANCE, MAX_PREVIEW_ROWS

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_SYSTEM = (
    """\
You are a data summarizer for a US rent burden chatbot.

Summarize the query results in 1-3 concise sentences.
Rules:
- Use ONLY the numbers and facts present in the result rows below.
- Do NOT fabricate statistics or reference data outside what is shown.
- Be specific: include key values, percentages, and location names from the data.
- If there are multiple rows, identify the most notable finding (highest,
  lowest, largest, etc.) and mention it explicitly.

"""
    + LLM_POPULATION_GUIDANCE
)

_USER_TMPL = """\
Question: {question}
Source table(s): {tables}
SQL used: {sql}
Result rows (up to {preview_rows} shown of {row_count} total):
{rows_formatted}
"""


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

def call_summarize_results(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    source_tables: List[str],
    sql_text: str,
    result_rows: List[Dict[str, Any]],
) -> str:
    """
    Use the Cortex LLM to produce a brief plain-language summary of query results.

    The LLM is shown only the returned rows and must summarize strictly from them.
    Returns a plain string (not a FinalAnswer) — intended as a lighter alternative
    to call_qa_agent when only a text snippet is needed.
    """
    rows_formatted = _format_rows(result_rows, MAX_PREVIEW_ROWS)
    user_prompt = _USER_TMPL.format(
        question=user_question,
        tables=", ".join(source_tables),
        sql=sql_text,
        preview_rows=MAX_PREVIEW_ROWS,
        row_count=len(result_rows),
        rows_formatted=rows_formatted,
    )
    answer = client.call_cortex(_SYSTEM, user_prompt)
    return answer.strip() or "No summary could be generated."
