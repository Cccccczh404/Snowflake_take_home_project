from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import SnowflakeAgentClient
from .config import LLM_POPULATION_GUIDANCE
from .models import FinalAnswer

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_SYSTEM = (
    """\
You are a friendly assistant for a US population rent burden chatbot.

The system could not complete the request. Your job is to:
1. Briefly explain the failure reason in plain, non-technical language. Do not
   lead with an apology or a blanket refusal unless that fits the situation.
2. Suggest 1-2 specific follow-up questions the user could ask next, tailored
   to where the failure occurred and what the conversation has covered so far.
   Good follow-ups: clarify a location, try a nearby geography, switch from
   county to state level, or rephrase a rent-burden question.
3. Never answer, speculate, or discuss topics unrelated to US population or
   rent burden. If you would need off-topic content to be helpful, stay silent
   on that and redirect to rent burden or population statistics only.

If the request violates the safety policy, say you cannot answer because of the
safety policy — do not justify with internal detail.

If the request was irrelevant to this assistant from the start, say that you
are for US rent burden and related population statistics, and suggest what you
can help with (for example: rent burden by state, county, or neighborhood).

Keep your response to 2-4 sentences. Do NOT mention SQL, stored procedures,
error codes, or any internal system details.

"""
    + LLM_POPULATION_GUIDANCE
)

_USER_TMPL = """\
{history_block}Original question: {question}
Pipeline stage where failure occurred: {status}
Failure reason: {error_message}
Additional context: {context}
"""

_HISTORY_TMPL = """\
[RECENT CONVERSATION — last {n} turn(s)]
{turns}
[END HISTORY]

"""


def _build_history_block(history: List[dict]) -> str:
    if not history:
        return ""
    lines = []
    for i, turn in enumerate(history, 1):
        lines.append(f"Turn {i}: User: {turn['user']}")
        lines.append(f"        Bot : {turn['bot'][:200]}{'…' if len(turn['bot']) > 200 else ''}")
    return _HISTORY_TMPL.format(n=len(history), turns="\n".join(lines))


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def call_exception_agent(
    client: SnowflakeAgentClient,
    conversation_id: str,
    user_question: str,
    status: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    history: Optional[List[dict]] = None,
) -> FinalAnswer:
    """
    Use the Cortex LLM to produce a friendly, plain-language error response
    with contextual follow-up suggestions.

    Receives the failure status (pipeline stage), error message, and the last
    few conversation turns so follow-up suggestions are grounded in what the
    user was already trying to do.

    Always returns FinalAnswer with ok=False.
    """
    history_block = _build_history_block(history or [])
    context_str = str(context or {})
    user_prompt = _USER_TMPL.format(
        history_block=history_block,
        question=user_question,
        status=status,
        error_message=error_message,
        context=context_str,
    )
    user_message = client.call_cortex(_SYSTEM, user_prompt)

    return FinalAnswer(
        ok=False,
        status=status,
        user_message=user_message.strip(),
        route="exception_agent",
        debug={
            "error_message": error_message,
            "context": context or {},
        },
    )
