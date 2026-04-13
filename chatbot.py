from __future__ import annotations

import json
import sys
import uuid
import traceback

# Set to False to silence step-by-step debug output
DEBUG = True


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"  [DEBUG] {msg}", flush=True)

from typing import Optional, Tuple

from utils import (
    SNOWFLAKE_CONNECTION,
    MAX_PREVIEW_ROWS,
    SnowflakeAgentClient,
    FinalAnswer,
    QueryExecutionResult,
    TableSelection,
    call_intent_router,
    call_exception_agent,
    call_pick_tables,
    call_generate_sql,
    call_validate_sql,
    call_qa_agent,
    synthesize_response,
)


# ============================================================
# ORCHESTRATOR
# Flow:
#   user question
#     → [agent_intent_router]  classify relevance
#     → [tool_pick_tables]     select right table(s)
#     → [tool_generate_sql]    generate constrained SELECT
#     → [tool_validate_sql]    pure-Python safety checks
#     → query_rows()           run the query
#     → [agent_qa]             ground answer in returned rows
#   any failure → [agent_exception]  friendly natural-language error
# ============================================================

# Simple affirmative / negative word sets for clarification confirmation
_AFFIRMATIVES = {"yes", "yeah", "yep", "yup", "correct", "right", "sure",
                 "ok", "okay", "confirm", "confirmed", "that's right", "go ahead"}
_NEGATIVES    = {"no", "nope", "nah", "wrong", "incorrect", "cancel", "never mind"}

HISTORY_LIMIT = 3  # how many past turns to carry


class PopulationChatbot:
    def __init__(self, client: SnowflakeAgentClient) -> None:
        self.client = client
        self.conversation_id = str(uuid.uuid4())
        # Rolling window of the last HISTORY_LIMIT turns: [{"user": ..., "bot": ...}]
        self.history: list = []
        # Set when a NEEDS_CLARIFICATION response was issued; cleared on resolution
        self._pending: Optional[dict] = None  # {"original": str, "resolved": str, "items": list}

    def _record(self, user_input: str, agent_response: str) -> None:
        """Append a turn to history, capping at HISTORY_LIMIT."""
        self.history.append({"user": user_input, "bot": agent_response})
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)

    def _generate_and_run(
        self,
        cid: str,
        user_question: str,
        table_sel: TableSelection,
        history: Optional[list] = None,
    ) -> Tuple[Optional[QueryExecutionResult], Optional[str], Optional[dict]]:
        """
        Steps 3–5: generate SQL → validate → execute, with one automatic retry.

        On the first failure (generation, validation, or execution) the error
        is fed back to call_generate_sql so it can produce a corrected query.
        If the second attempt also fails, returns (None, None, error_dict).
        On success, returns (QueryExecutionResult, final_sql, None).
        """
        previous_sql: Optional[str] = None
        previous_error: Optional[str] = None
        hist = history or []

        for attempt in range(1, 3):  # attempt 1, then attempt 2
            _dbg(f"[attempt {attempt}] generating SQL for tables: {table_sel.tables}")

            # Step 3 — Generate SQL
            sql_gen = call_generate_sql(
                client=self.client,
                conversation_id=cid,
                user_question=user_question,
                tables=table_sel.tables,
                grain=table_sel.grain,
                previous_sql=previous_sql,
                previous_error=previous_error,
                history=hist,
            )

            if not sql_gen.ok or not sql_gen.sql:
                err_msg = sql_gen.message or "Could not generate a query for this question."
                _dbg(f"[attempt {attempt}] SQL GENERATION FAILED: {err_msg}")
                if attempt == 1:
                    previous_error = f"SQL generation failed: {err_msg}"
                    continue
                return None, None, {
                    "status": "SQL_GENERATION_FAILED",
                    "message": err_msg,
                    "context": {"tables": table_sel.tables, "attempt": attempt},
                }

            _dbg(f"[attempt {attempt}] generated SQL:\n  {sql_gen.sql}")

            # Step 4 — Validate SQL (LLM-based)
            sql_val = call_validate_sql(
                client=self.client,
                conversation_id=cid,
                user_question=user_question,
                sql_text=sql_gen.sql,
                tables=table_sel.tables,
            )

            if not sql_val.ok:
                _dbg(f"[attempt {attempt}] SQL VALIDATION FAILED: {sql_val.message}")
                if attempt == 1:
                    previous_sql = sql_gen.sql
                    previous_error = f"SQL validation failed: {sql_val.message}"
                    continue
                return None, None, {
                    "status": "SQL_VALIDATION_FAILED",
                    "message": sql_val.message,
                    "context": {"generated_sql": sql_gen.sql, "tables": table_sel.tables},
                }

            final_sql = sql_val.rewritten_sql or sql_gen.sql
            if sql_val.rewritten_sql:
                _dbg(f"[attempt {attempt}] validator rewrote SQL:\n  {final_sql}")

            # Step 5 — Execute
            _dbg(f"[attempt {attempt}] executing SQL...")
            query_result = self.client.query_rows(final_sql)

            if not query_result.ok:
                _dbg(f"[attempt {attempt}] EXECUTION FAILED: {query_result.message}")
                if attempt == 1:
                    previous_sql = final_sql
                    previous_error = f"Query execution failed: {query_result.message}"
                    continue
                return None, None, {
                    "status": "QUERY_EXECUTION_FAILED",
                    "message": query_result.message,
                    "context": {"sql": final_sql, "tables": table_sel.tables, "attempts": 2},
                }

            _dbg(f"[attempt {attempt}] query OK — {query_result.row_count} row(s) returned.")

            return query_result, final_sql, None

        # Should not be reached, but satisfies type checker
        return None, None, {"status": "UNKNOWN_ERROR", "message": "Retry loop exhausted."}

    def handle_user_question(self, user_question: str) -> FinalAnswer:
        cid = self.conversation_id
        try:
            # ------------------------------------------------
            # Step 0 — Pending clarification resolution
            # If we previously asked "did you mean X?", handle yes/no
            # before doing anything else.
            # ------------------------------------------------
            if self._pending:
                low = user_question.strip().lower()
                if low in _AFFIRMATIVES:
                    # User confirmed — swap in the resolved question
                    resolved_question = self._pending["resolved"]
                    _dbg(f"clarification confirmed → using: {resolved_question}")
                    self._pending = None
                    resp = self._run_qa_pipeline(cid, user_question, resolved_question, history=self.history)
                    self._record(user_question, resp.user_message)
                    return resp
                elif low in _NEGATIVES:
                    self._pending = None
                    resp = FinalAnswer(
                        ok=False,
                        status="CLARIFICATION_DECLINED",
                        user_message=(
                            "No problem! Please rephrase your question using the full "
                            "state or county name and I'll look it up for you."
                        ),
                        route="clarification",
                    )
                    self._record(user_question, resp.user_message)
                    return resp
                else:
                    # Not a yes/no — treat as a fresh question, clear pending
                    self._pending = None

            # ------------------------------------------------
            # Step 1 — Intent routing
            # Pass conversation history so the router understands follow-ups.
            # ------------------------------------------------
            route = call_intent_router(
                client=self.client,
                conversation_id=cid,
                user_question=user_question,
                history=self.history,
            )

            if route.route == "conversational_agent":
                # ------------------------------------------------
                # Conversational/meta question — no data query needed.
                # Pass straight to the response synthesizer with history
                # so it can answer from conversation context alone.
                # ------------------------------------------------
                _dbg(f"conversational route — intent: {route.intent}")
                stub = FinalAnswer(
                    ok=True,
                    status="CONVERSATIONAL",
                    user_message="",
                    route="conversational_agent",
                )
                resp = synthesize_response(
                    client=self.client,
                    user_question=user_question,
                    answer=stub,
                    history=self.history,
                    dialogue=True,
                )
                resp.route = "conversational_agent"
                resp.status = "CONVERSATIONAL"
                self._record(user_question, resp.user_message)
                return resp

            if route.route != "qa_agent":
                resp = call_exception_agent(
                    client=self.client,
                    conversation_id=cid,
                    user_question=user_question,
                    status="OUT_OF_SCOPE",
                    error_message=route.reason or "Question is outside the supported scope.",
                    context={"intent": route.intent},
                )
                self._record(user_question, resp.user_message)
                return resp

            # ------------------------------------------------
            # Step 1b — Offline fuzzy location resolution
            # HIGH confidence → silently inject canonical name into question.
            # MEDIUM confidence → stop and ask the user to confirm before
            #   proceeding, so we never query with a wrong location.
            # ------------------------------------------------
            resolved_question, clarifications = self.client.augment_question(user_question)

            if clarifications:
                # MEDIUM-confidence match — ask user to confirm before querying
                lines = [
                    "I want to make sure I understood the location correctly. "
                    "Did you mean:"
                ]
                for original, canonical, loc_type in clarifications:
                    lines.append(f'  • "{original}" → {canonical} ({loc_type})?')
                lines.append(
                    '\nJust say "yes" to confirm, or rephrase with the full name.'
                )
                msg = "\n".join(lines)

                # Build the fully-resolved question to use when confirmed.
                # MEDIUM matches are not yet in resolved_question (only HIGH are),
                # so we append them explicitly here as confirmed context.
                medium_ctx = ", ".join(
                    f"{canonical} ({loc_type})"
                    for _, canonical, loc_type in clarifications
                )
                if "[location context:" in resolved_question:
                    confirmed_resolved = resolved_question.rstrip("]") + f"; {medium_ctx}]"
                else:
                    confirmed_resolved = user_question + f"  [location context: {medium_ctx}]"

                self._pending = {
                    "original": user_question,
                    "resolved": confirmed_resolved,
                    "items": clarifications,
                }
                resp = FinalAnswer(
                    ok=False,
                    status="NEEDS_CLARIFICATION",
                    user_message=msg,
                    route="clarification",
                )
                self._record(user_question, msg)
                return resp

            if resolved_question != user_question:
                _dbg(f"location resolved: {resolved_question}")

            resp = self._run_qa_pipeline(cid, user_question, resolved_question, route.intent, self.history)
            self._record(user_question, resp.user_message)
            return resp

        except Exception as exc:
            _dbg(f"UNHANDLED_EXCEPTION: {exc}\n{traceback.format_exc()}")
            resp = call_exception_agent(
                client=self.client,
                conversation_id=self.conversation_id,
                user_question=user_question,
                status="UNHANDLED_EXCEPTION",
                error_message=str(exc),
                context={"traceback": traceback.format_exc()},
            )
            self._record(user_question, resp.user_message)
            return resp

    def _run_qa_pipeline(
        self,
        cid: str,
        display_question: str,
        resolved_question: str,
        intent: Optional[str] = None,
        history: Optional[list] = None,
    ) -> FinalAnswer:
        """
        Steps 2–7: table selection → SQL gen/validate/execute → QA → synthesis.
        display_question — original user text (shown in error messages)
        resolved_question — fuzzy-augmented text (used for all LLM calls)
        history         — last HISTORY_LIMIT turns passed to each LLM agent
        """
        hist = history or []

        # Step 2 — Table selection
        table_sel = call_pick_tables(
            client=self.client,
            conversation_id=cid,
            user_question=resolved_question,
            intent=intent,
            history=hist,
        )

        if not table_sel.ok:
            return call_exception_agent(
                client=self.client,
                conversation_id=cid,
                user_question=display_question,
                status="MISSING_REQUIRED_INFO",
                error_message=(
                    table_sel.message
                    or f"Need more information to answer. Missing: {table_sel.filters_needed}"
                ),
                context={"filters_needed": table_sel.filters_needed},
            )

        if not table_sel.tables:
            return call_exception_agent(
                client=self.client,
                conversation_id=cid,
                user_question=display_question,
                status="TABLE_SELECTION_FAILED",
                error_message="Could not identify which dataset to query.",
                context={"table_sel_message": table_sel.message},
            )

        # Steps 3–5 — Generate → Validate → Execute (with retry)
        query_result, final_sql, last_error = self._generate_and_run(
            cid=cid,
            user_question=resolved_question,
            table_sel=table_sel,
            history=hist,
        )

        if last_error:
            return call_exception_agent(
                client=self.client,
                conversation_id=cid,
                user_question=display_question,
                status=last_error["status"],
                error_message=last_error["message"],
                context=last_error.get("context", {}),
            )

        if query_result.row_count == 0:
            return call_exception_agent(
                client=self.client,
                conversation_id=cid,
                user_question=display_question,
                status="NO_RESULTS",
                error_message=(
                    "The query ran successfully but returned no matching rows. "
                    "The location or filter you specified may not exist in the dataset."
                ),
                context={"sql": final_sql, "tables": table_sel.tables},
            )

        # Step 6 — Grounded QA response
        final_answer = call_qa_agent(
            client=self.client,
            conversation_id=cid,
            user_question=resolved_question,
            source_tables=table_sel.tables,
            sql_text=final_sql,
            query_result=query_result,
            history=hist,
        )
        final_answer.filters = {"grain": table_sel.grain, "intent": intent}

        # Step 7 — Response synthesis (tone polish + follow-ups, with history)
        final_answer = synthesize_response(
            client=self.client,
            user_question=resolved_question,
            answer=final_answer,
            history=self.history,
        )
        return final_answer


# ============================================================
# CLI
# ============================================================

def print_response(resp: FinalAnswer) -> None:
    width = 100
    print("\n" + "=" * width)
    if resp.route == "clarification":
        print(f"[?] CLARIFICATION NEEDED")
        print("-" * width)
        print(resp.user_message)
        print("=" * width + "\n")
        return
    if resp.route == "conversational_agent":
        print(f"[CHAT]")
        print("-" * width)
        print(resp.user_message)
        print("=" * width + "\n")
        return
    status_icon = "OK" if resp.ok else "ERR"
    print(f"[{status_icon}] STATUS: {resp.status}   ROUTE: {resp.route}")
    print("-" * width)
    print(resp.user_message)

    if resp.source_tables:
        print("\nSource table(s):")
        for t in resp.source_tables:
            print(f"  {t}")

    if resp.sql:
        print("\nSQL used:")
        print(f"  {resp.sql}")

    if resp.filters:
        print("\nContext:")
        print(f"  {json.dumps(resp.filters)}")

    if resp.data:
        shown = min(len(resp.data), MAX_PREVIEW_ROWS)
        print(f"\nData ({shown} of {resp.row_count} row(s)):")
        for row in resp.data[:MAX_PREVIEW_ROWS]:
            print(f"  {json.dumps(row, default=str)}")

    print("=" * width + "\n")


def main() -> None:
    client = SnowflakeAgentClient(SNOWFLAKE_CONNECTION)
    bot = PopulationChatbot(client)

    print("US Population Rent Burden Chatbot")
    print("Ask about rent burden by state or county. Type 'exit' to quit.\n")

    try:
        while True:
            try:
                user_question = input("You: ").strip()
            except EOFError:
                break
            if not user_question:
                continue
            if user_question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            response = bot.handle_user_question(user_question)
            print_response(response)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
