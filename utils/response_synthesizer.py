from __future__ import annotations

from typing import List, Optional

from .client import SnowflakeAgentClient
from .config import LLM_POPULATION_GUIDANCE
from .models import FinalAnswer, LocationContext

# ----------------------------------------------------------
# Prompts
# ----------------------------------------------------------

_SYSTEM = (
    """\
You are a friendly response synthesizer for a US rent burden chatbot.

You will receive a raw data answer that was grounded strictly in database results.
Your job is to:
1. Keep every number, percentage, state name, county name, and fact EXACTLY as written.
   Do NOT round, alter, omit, or paraphrase any data value.
2. Rewrite the answer in a warm, clear, conversational tone — as if a knowledgeable
   analyst is speaking directly to the user.
3. At the end, add a short friendly closing line such as
   "Is there anything else I can help you with?"
   AND suggest 2-3 short relevant follow-up questions the user might find interesting,
   based on the topic of the answer (e.g. comparisons, trends, rankings).

Format your response as:

<answer>
[your rewritten answer with all data intact]
</answer>

<followups>
- [follow-up question 1]
- [follow-up question 2]
- [follow-up question 3]
</followups>

Rules:
- Never invent data not present in the raw answer.
- Never drop or change a number, ratio, or location name.
- Keep the answer concise — do not add filler paragraphs.
- The follow-up questions must be answerable from the same dataset
  (US rent burden data by state, county, or census block group).
- DIALOGUE MODE: If the raw data answer section is empty or absent, that means
  this is a purely conversational message — no database query was run. In that
  case you MUST answer based solely on the conversation history provided above.
  Do NOT introduce any outside knowledge or invent data. If the history does not
  contain enough context to answer, say so honestly and invite the user to ask a
  data question.

"""
    + LLM_POPULATION_GUIDANCE
)

_USER_TMPL = """\
{history_block}Original user question: {question}

Raw data answer to rewrite:
{raw_answer}
"""

_USER_DIALOGUE_TMPL = """\
{history_block}{loc_block}User message: {question}

[No database query was run — this is a conversational message.
Answer using only the conversation history and active location state above.]
"""

_HISTORY_BLOCK_TMPL = """\
[RECENT CONVERSATION — last {n} turn(s)]
{turns}
[END HISTORY]

"""


def _parse_sections(text: str) -> tuple[str, List[str]]:
    """
    Extract <answer>...</answer> and <followups>...</followups> from the LLM output.
    Falls back gracefully if the tags are missing.
    """
    import re

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    followups_match = re.search(r"<followups>(.*?)</followups>", text, re.DOTALL | re.IGNORECASE)

    answer = answer_match.group(1).strip() if answer_match else text.strip()

    followups: List[str] = []
    if followups_match:
        raw_followups = followups_match.group(1).strip()
        for line in raw_followups.splitlines():
            line = line.lstrip("-•* ").strip()
            if line:
                followups.append(line)

    return answer, followups


# ----------------------------------------------------------
# Public function
# ----------------------------------------------------------

def _build_history_block(history: list) -> str:
    if not history:
        return ""
    turns = "".join(
        f"  User : {e['user']}\n  Agent: {e['bot'][:300]}\n\n"
        for e in history
    )
    return _HISTORY_BLOCK_TMPL.format(n=len(history), turns=turns)


def synthesize_response(
    client: SnowflakeAgentClient,
    user_question: str,
    answer: FinalAnswer,
    history: list | None = None,
    dialogue: bool = False,
    location_ctx: Optional[LocationContext] = None,
) -> FinalAnswer:
    """
    Post-process a FinalAnswer through the response synthesizer.

    When dialogue=True the user prompt is built without a raw data answer,
    instructing the LLM to reply purely from conversation history.
    Only user_message is modified — all data fields pass through unchanged.
    """
    history_block = _build_history_block(history or [])
    if dialogue:
        loc_block = location_ctx.as_prompt_block() if location_ctx and location_ctx.is_set() else ""
        user_prompt = _USER_DIALOGUE_TMPL.format(
            history_block=history_block,
            loc_block=loc_block,
            question=user_question,
        )
    else:
        user_prompt = _USER_TMPL.format(
            history_block=history_block,
            question=user_question,
            raw_answer=answer.user_message,
        )
    raw = client.call_cortex(_SYSTEM, user_prompt)
    synthesized_answer, followups = _parse_sections(raw)

    # Build the final user-facing message
    parts = [synthesized_answer]
    if followups:
        parts.append("\nYou might also want to ask:")
        for q in followups:
            parts.append(f"  • {q}")

    answer.user_message = "\n".join(parts)
    return answer
