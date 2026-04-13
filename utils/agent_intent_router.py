from __future__ import annotations

import json

from .client import SnowflakeAgentClient, parse_llm_json
from .config import LLM_POPULATION_GUIDANCE
from .models import RouteResult

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_SYSTEM = (
    """\
You are an intent classifier for a US population rent burden chatbot.

The chatbot ONLY answers questions about US population and rent burden data:
- Rent burden ratios at the state, county, or census block group level
- Number of renters and how much of their income goes to rent
- Comparisons between states or counties
- Rankings (most/least burdened states, counties, etc.)

"""
    + LLM_POPULATION_GUIDANCE
    + """

Classify the user question into one of three routes:

1. "qa_agent"           — the question requires querying rent burden data to answer
                          (e.g. "What is the rent burden in California?",
                           "Which county has the highest burden?")

2. "conversational_agent" — the question is topically related to the conversation
                          or the rent burden domain, but does NOT require a database
                          query to answer. Use this for:
                          - Meta questions about the conversation ("which state did
                            I just ask about?", "what did you say earlier?")
                          - Greetings or closing phrases ("thanks", "great", "ok")
                          - Simple clarifications that can be answered from history
                          - Short affirmations or follow-up remarks in context
                          ONLY use this route when conversation history is present
                          and provides sufficient context; otherwise use
                          exception_agent.

3. "exception_agent"    — the question is irrelevant, off-topic, too vague with no
                          history context, or requests data that doesn't exist
                          (e.g. weather, sports, prices of goods, political topics).
                          Missing location only applies when the question needs a
                          sub-national place and none is implied — not for whole-US
                          phrases ("us", "US", nationwide), which are answerable via
                          STATE_SUMMARY.

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{
  "route": "qa_agent" | "conversational_agent" | "exception_agent",
  "reason": "one-sentence explanation of why you chose this route",
  "intent": "short_snake_case_intent_label or null"
}

Intent label examples: compare_state_burden, find_most_burdened_county,
get_state_statistics, lookup_county_burden, rank_states_by_severe_burden,
us_national_population_or_renters, other_data_question,
conversational_followup, meta_question_about_context
"""
)

_USER_TMPL = """\
{history_block}Question: {question}"""

_HISTORY_HEADER = """\
[CONVERSATION HISTORY — last {n} turn(s), oldest first]
{turns}
[END HISTORY]

"""

_TURN_TMPL = """\
Turn {i}:
  User : {user}
  Agent: {agent}
"""


def _build_history_block(history: list) -> str:
    """Format up to the last 3 (user, bot) turns for the prompt."""
    if not history:
        return ""
    turns = "".join(
        _TURN_TMPL.format(i=i + 1, user=entry["user"], agent=entry["bot"][:300])
        for i, entry in enumerate(history)
    )
    return _HISTORY_HEADER.format(n=len(history), turns=turns)


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def call_intent_router(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    history: list | None = None,
) -> RouteResult:
    """
    Use the Cortex LLM to classify whether the question belongs to the
    qa_agent flow or should be rejected by the exception_agent.

    history — list of dicts {"user": ..., "agent": ...} for the last 3 turns.
              Lets the router understand follow-ups like "yes", "what about TX?".
    """
    history_block = _build_history_block(history or [])
    user_prompt = _USER_TMPL.format(
        history_block=history_block,
        question=user_question,
    )
    raw = client.call_cortex(_SYSTEM, user_prompt)

    try:
        data = parse_llm_json(raw)
        return RouteResult(
            route=data.get("route", "exception_agent"),
            reason=data.get("reason", ""),
            intent=data.get("intent"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # If the LLM returns malformed JSON, fail safe to exception_agent
        return RouteResult(
            route="exception_agent",
            reason=f"Intent router returned unparseable response: {raw[:200]}",
            intent=None,
        )
