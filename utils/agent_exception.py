from __future__ import annotations

from typing import Any, Dict, Optional

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
2. Offer one or two on-topic follow-ups (for example: clarify a location, try a
   nearby geography, or rephrase a rent-burden question for state, county, or
   neighborhood).
3. Never answer, speculate, or discuss topics unrelated to US population or
   rent burden. If you would need off-topic content to be helpful, stay silent
   on that and redirect to rent burden or population statistics only.

If the request violates the safety policy, say you cannot answer because of the
safety policy—do not justify with internal detail.

If the request was irrelevant to this assistant from the start, say that you
are for US rent burden and related population statistics, and suggest what you
can help with (for example: rent burden by state, county, or neighborhood).

Keep your response to 2-4 sentences. Do NOT mention SQL, stored procedures,
error codes, or any internal system details.

"""
    + LLM_POPULATION_GUIDANCE
)

_USER_TMPL = """\
Original question: {question}
Issue: {error_message}
Status code: {status}
Additional context: {context}
"""


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
) -> FinalAnswer:
    """
    Use the Cortex LLM to produce a friendly, plain-language error response.

    Always returns FinalAnswer with ok=False so the orchestrator knows
    the request did not complete successfully.
    """
    context_str = str(context or {})
    user_prompt = _USER_TMPL.format(
        question=user_question,
        error_message=error_message,
        status=status,
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
