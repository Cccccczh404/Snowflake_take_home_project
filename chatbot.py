from __future__ import annotations

import json
import re
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
    LocationContext,
    LocationStateManager,
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
        # Explicit location state — updated by confirmed location mentions
        self._loc_mgr = LocationStateManager(client._location_resolver)
        # Set when a NEEDS_CLARIFICATION response was issued; cleared on resolution
        self._pending: Optional[dict] = None  # {"original": str, "items": list}

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
        location_ctx: Optional[LocationContext] = None,
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
                location_ctx=location_ctx,
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

    # --------------------------------------------------------
    # CBG disambiguation helpers
    # --------------------------------------------------------

    @staticmethod
    def _extract_county_hint(text: str) -> Optional[str]:
        """
        Extract a 'X County' or 'X Parish' phrase from free text.
        Returns the matched phrase (e.g. 'Washington County') or None.
        """
        m = re.search(
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:County|Parish|Borough|Census\s+Area)\b',
            text,
        )
        return m.group(0) if m else None

    def _check_cbg_county_ambiguity(self, county_hint: str) -> list:
        """
        Return the list of distinct STATE_NAME values in CBG_POPULATION_FEATURES
        that match the given county_hint.  Returns [] on query error.
        """
        safe = county_hint.replace("'", "''")
        result = self.client.query_rows(
            f"SELECT DISTINCT STATE_NAME FROM CBG_POPULATION_FEATURES "
            f"WHERE COUNTY_NAME LIKE '%{safe}%' ORDER BY STATE_NAME LIMIT 60"
        )
        if not result.ok or not result.rows:
            return []
        return [row["STATE_NAME"] for row in result.rows]

    def _maybe_cbg_disambiguate(
        self,
        display_question: str,
        resolved_question: str,
        table_sel: TableSelection,
        location_ctx: Optional[LocationContext],
    ) -> Optional[FinalAnswer]:
        """
        Called after table selection when grain == census_block_group.

        Returns a clarification FinalAnswer (and sets self._pending) if the
        county is ambiguous across multiple states.
        Returns None when no disambiguation is needed and the pipeline can proceed.

        Skipped entirely when:
          - grain is not census_block_group
          - a FIPS code is already in LocationContext (exact match, no ambiguity)
          - the state is already confirmed in LocationContext
          - no county hint can be extracted
          - the county matches exactly one state
        """
        if table_sel.grain != "census_block_group":
            return None

        # FIPS-based search: exact match, no ambiguity possible
        if location_ctx and (location_ctx.cbg_fips or location_ctx.tract_fips):
            return None

        # State already known: trust it, skip check
        if location_ctx and location_ctx.state:
            return None

        # Extract county hint: prefer LocationContext, fall back to question text
        county_hint: Optional[str] = None
        if location_ctx and location_ctx.county:
            county_hint = location_ctx.county.split(",")[0].strip()
        if not county_hint:
            county_hint = self._extract_county_hint(resolved_question)
        if not county_hint:
            return None  # No county to check

        matching_states = self._check_cbg_county_ambiguity(county_hint)
        _dbg(f"CBG county ambiguity check for '{county_hint}': {matching_states}")

        if len(matching_states) <= 1:
            return None  # Unique (or 0 matches) — proceed

        # Multiple states — ask the user to pick one
        shown = matching_states[:12]
        lines = [
            f'"{county_hint}" exists in multiple states. Which state did you mean?'
        ]
        for s in shown:
            lines.append(f"  • {s}")
        if len(matching_states) > 12:
            lines.append(f"  (and {len(matching_states) - 12} more — type the full state name)")
        lines.append("\nJust type the state name and I'll search that county for you.")

        msg = "\n".join(lines)
        self._pending = {
            "type": "cbg_disambiguation",
            "original": display_question,
            "resolved": resolved_question,
            "county_hint": county_hint,
            "matching_states": matching_states,
        }
        return FinalAnswer(
            ok=False,
            status="NEEDS_CLARIFICATION",
            user_message=msg,
            route="clarification",
        )

    def handle_user_question(self, user_question: str) -> FinalAnswer:
        cid = self.conversation_id
        try:
            # ------------------------------------------------
            # Step 0 — Pending clarification resolution
            # If we previously asked "did you mean X?", handle yes/no
            # before doing anything else.
            # ------------------------------------------------
            if self._pending:
                ptype = self._pending.get("type", "fuzzy")
                low = user_question.strip().lower()

                # ---- CBG county-disambiguation pending ----
                if ptype == "cbg_disambiguation":
                    # User is responding with a state name (e.g. "Pennsylvania")
                    matching_states = self._pending["matching_states"]
                    selected_state: Optional[str] = None

                    # Try direct fuzzy resolution first
                    fuzz_result = self._loc_mgr._resolver.resolve_token(user_question.strip())
                    if fuzz_result and fuzz_result[0] == "state":
                        candidate = fuzz_result[1]
                        if candidate in matching_states:
                            selected_state = candidate

                    # Fall back: substring match against matching_states list
                    if not selected_state:
                        for s in matching_states:
                            if low in s.lower() or s.lower().startswith(low):
                                selected_state = s
                                break

                    if selected_state:
                        original_q = self._pending["original"]
                        self._pending = None
                        self._loc_mgr.confirm_medium("state", selected_state)
                        resolved_q = self._loc_mgr.annotate_from_state(original_q)
                        _dbg(f"CBG state selected: {selected_state}  ctx: {self._loc_mgr.ctx}")
                        resp = self._run_qa_pipeline(
                            cid, original_q, resolved_q,
                            history=self.history,
                            location_ctx=self._loc_mgr.ctx,
                        )
                        self._record(user_question, resp.user_message)
                        return resp
                    else:
                        # Can't match — ask again with the list
                        shown = matching_states[:12]
                        options = ", ".join(shown)
                        if len(matching_states) > 12:
                            options += f" (and {len(matching_states) - 12} more)"
                        resp = FinalAnswer(
                            ok=False,
                            status="NEEDS_CLARIFICATION",
                            user_message=(
                                f"I couldn't match '{user_question}' to a state. "
                                f"Please type one of: {options}"
                            ),
                            route="clarification",
                        )
                        self._record(user_question, resp.user_message)
                        return resp

                # ---- Fuzzy location clarification pending (yes/no) ----
                if low in _AFFIRMATIVES:
                    # User confirmed — update LocationContext, rebuild resolved question
                    for _orig, canonical, loc_type in self._pending["items"]:
                        self._loc_mgr.confirm_medium(loc_type, canonical)
                    original_q = self._pending["original"]
                    resolved_question = self._loc_mgr.annotate_from_state(original_q)
                    _dbg(f"clarification confirmed → ctx: {self._loc_mgr.ctx}")
                    _dbg(f"resolved question: {resolved_question}")
                    self._pending = None
                    resp = self._run_qa_pipeline(
                        cid, original_q, resolved_question,
                        history=self.history,
                        location_ctx=self._loc_mgr.ctx,
                    )
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
                # Conversational/meta question — no data query.
                # Pass LocationContext + history to the synthesizer so it
                # can answer questions like "which state was I asking about?".
                # Fuzzy search does NOT run for conversational messages.
                # ------------------------------------------------
                _dbg(f"conversational route — intent: {route.intent}, loc_ctx: {self._loc_mgr.ctx}")
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
                    location_ctx=self._loc_mgr.ctx,
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
            # Step 1b — Location state resolution (qa_agent path only)
            # LocationStateManager scans the message, updates LocationContext
            # for HIGH matches, and carries forward state for follow-ups.
            # MEDIUM matches → ask the user to confirm; no state change yet.
            # ------------------------------------------------
            resolved_question, clarifications = self._loc_mgr.process_message(user_question)

            if clarifications:
                # MEDIUM-confidence match — ask user to confirm before querying.
                # Store original question + match items; state is NOT updated yet.
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
                self._pending = {
                    "original": user_question,
                    "items": clarifications,  # (original_token, canonical, loc_type)
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
                _dbg(f"location resolved: {resolved_question}  ctx: {self._loc_mgr.ctx}")

            resp = self._run_qa_pipeline(
                cid, user_question, resolved_question, route.intent,
                self.history, self._loc_mgr.ctx,
            )
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
        location_ctx: Optional[LocationContext] = None,
    ) -> FinalAnswer:
        """
        Steps 2–7: table selection → SQL gen/validate/execute → QA → synthesis.
        display_question — original user text (shown in error messages)
        resolved_question — state-annotated text (used for all LLM calls)
        history          — last HISTORY_LIMIT turns passed to each LLM agent
        location_ctx     — current LocationContext for canonical name injection
        """
        hist = history or []

        # Step 2 — Table selection
        table_sel = call_pick_tables(
            client=self.client,
            conversation_id=cid,
            user_question=resolved_question,
            intent=intent,
            history=hist,
            location_ctx=location_ctx,
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

        # Step 2b — CBG county disambiguation (only for census_block_group grain)
        # Checks whether the county name is ambiguous across multiple states and,
        # if so, pauses and asks the user to specify the state.
        # This check is skipped when a FIPS code or confirmed state is already set.
        disambig = self._maybe_cbg_disambiguate(
            display_question, resolved_question, table_sel, location_ctx
        )
        if disambig is not None:
            return disambig

        # Steps 3–5 — Generate → Validate → Execute (with retry)
        query_result, final_sql, last_error = self._generate_and_run(
            cid=cid,
            user_question=resolved_question,
            table_sel=table_sel,
            history=hist,
            location_ctx=location_ctx,
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
            location_ctx=location_ctx,
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
