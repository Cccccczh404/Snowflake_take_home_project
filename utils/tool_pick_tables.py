from __future__ import annotations

import json
import re
from typing import Optional

from .client import SnowflakeAgentClient, parse_llm_json
from .config import LLM_POPULATION_GUIDANCE, TABLE_CATALOG
from .models import LocationContext, TableSelection

# ----------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------

def _build_catalog_text() -> str:
    lines = []
    for table_name, info in TABLE_CATALOG.items():
        lines.append(f"Table: {table_name}")
        lines.append(f"  Description : {info['description']}")
        lines.append(f"  Grain       : {info['grain']}")
        lines.append(f"  Use when    : {info['use_when']}")
        lines.append(f"  Columns:")
        for col_name, col_val in info["columns"].items():
            col_desc = col_val[1] if isinstance(col_val, tuple) else col_val
            lines.append(f"    - {col_name}: {col_desc}")
        lines.append("")
    return "\n".join(lines)


_CATALOG_TEXT = _build_catalog_text()

# Strong US-national + population phrasing → STATE_SUMMARY without calling the LLM
# (avoids false "too vague" when scope is the whole country).
_NATIONAL_US_POPULATION = re.compile(
    r"(?:^|\s)(?:u\.s\.|us)\s+population\b|"
    r"\bpopulation\s+(?:of|in)\s+(?:the\s+)?(?:u\.s\.|us|united\s+states)\b|"
    r"\b(?:u\.s\.|us|united\s+states)\s+population\b|"
    r"\bnational\s+population\b|"
    r"\bpopulation\s+of\s+(?:the\s+)?(?:whole\s+)?(?:country|nation)\b|"
    r"\b(?:whole|entire)\s+(?:u\.s\.|us)\s+population\b",
    re.IGNORECASE,
)


def _heuristic_national_us_tables(user_question: str) -> Optional[TableSelection]:
    q = user_question.strip()
    if not q or not _NATIONAL_US_POPULATION.search(q):
        return None
    return TableSelection(
        ok=True,
        tables=["STATE_SUMMARY"],
        grain="state",
        filters_needed=[],
        message="",
    )


_SYSTEM = f"""\
You are a table selector for a US rent burden database.

Available tables:
{_CATALOG_TEXT}

{LLM_POPULATION_GUIDANCE}

Rules:
- Prefer STATE_SUMMARY for state-level or national-level questions.
- Prefer COUNTY_SUMMARY for county-level or within-state county questions.
- Use CBG_POPULATION_FEATURES only when the question explicitly asks about
  neighborhoods, census block groups, or tracts.
- Only select multiple tables if the question truly requires joining them
  (this is rare — prefer a single table).
- Whole-country questions (United States / "US" / "us" / nationwide / "the
  country" / national population or renter totals) have a fully specified
  geography: set "ok" to true, tables ["STATE_SUMMARY"], grain "state",
  filters_needed [] — the answer is SUM across all state rows (never ask for
  a finer location for these).
- If the question is too vague (e.g. "what is the rent burden?" with no
  location AND not a national/whole-US scope as above), set "ok" to false and
  list what information is missing in "filters_needed".
- Do NOT include filters_needed entries for information that is already
  implied by the question or that the SQL can derive (e.g. if user says
  "all states", that is not missing info).

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "ok": true | false,
  "tables": ["TABLE_NAME"],
  "grain": "state" | "county" | "census_block_group",
  "filters_needed": [],
  "message": "explanation only when ok is false, empty string otherwise"
}}
"""

_USER_TMPL = """\
Intent: {intent}
Question: {question}
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

def call_pick_tables(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    intent: Optional[str],
    history: Optional[list] = None,
    location_ctx: Optional[LocationContext] = None,
) -> TableSelection:
    """
    Use the Cortex LLM to select the right Snowflake table(s) for the question.

    Returns TableSelection with ok=True and the chosen table names, or ok=False
    with filters_needed listing what information the user must provide.
    """
    shortcut = _heuristic_national_us_tables(user_question)
    if shortcut is not None:
        return shortcut

    history_block = _build_history_block(history or [])
    loc_block = location_ctx.as_prompt_block() if location_ctx and location_ctx.is_set() else ""
    user_prompt = history_block + loc_block + _USER_TMPL.format(
        intent=intent or "unknown",
        question=user_question,
    )
    raw = client.call_cortex(_SYSTEM, user_prompt)

    try:
        data = parse_llm_json(raw)
        return TableSelection(
            ok=bool(data.get("ok", False)),
            tables=data.get("tables", []),
            grain=data.get("grain"),
            filters_needed=data.get("filters_needed", []),
            message=data.get("message", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return TableSelection(
            ok=False,
            tables=[],
            grain=None,
            filters_needed=[],
            message=f"Table selector returned unparseable response: {raw[:200]}",
        )
