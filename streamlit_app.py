from __future__ import annotations

import os
from typing import Optional

import streamlit as st

def _hydrate_env_from_streamlit_secrets() -> None:
    """
    Streamlit Community Cloud provides secrets via `st.secrets`, not necessarily
    as process environment variables. The rest of the code reads Snowflake
    settings from env via `utils.config`, so we mirror secrets into env early.
    """
    try:
        secrets = st.secrets  # can raise if not running under Streamlit
    except Exception:
        return

    keys = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_AUTHENTICATOR",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    # Support both:
    # 1) flat secrets: SNOWFLAKE_ACCOUNT="..."
    # 2) sectioned secrets:
    #    [snowflake]
    #    SNOWFLAKE_ACCOUNT="..."
    snowflake_section = None
    try:
        snowflake_section = secrets.get("snowflake")
    except Exception:
        snowflake_section = None

    for k in keys:
        if os.getenv(k):
            continue
        if k in secrets and str(secrets[k]).strip():
            os.environ[k] = str(secrets[k])
            continue
        if isinstance(snowflake_section, dict) and k in snowflake_section and str(snowflake_section[k]).strip():
            os.environ[k] = str(snowflake_section[k])


_hydrate_env_from_streamlit_secrets()


import chatbot  # noqa: E402
from chatbot import PopulationChatbot  # noqa: E402
from utils import SNOWFLAKE_CONNECTION, SnowflakeAgentClient, FinalAnswer  # noqa: E402


APP_TITLE = "US Population Rent Burden Chatbot"


def _set_debug(enabled: bool) -> None:
    # Keep original behavior; just allow UI to toggle printing to logs.
    chatbot.DEBUG = bool(enabled)


@st.cache_resource(show_spinner="Starting chatbot…")
def _create_client() -> SnowflakeAgentClient:
    return SnowflakeAgentClient(SNOWFLAKE_CONNECTION)


def _get_bot() -> PopulationChatbot:
    if "bot" not in st.session_state:
        client = _create_client()
        st.session_state["bot"] = PopulationChatbot(client)
    return st.session_state["bot"]


def _render_answer(resp: FinalAnswer) -> None:
    st.markdown(resp.user_message)

    with st.expander("Details", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Route", resp.route)
        cols[1].metric("Status", resp.status)
        cols[2].metric("OK", "yes" if resp.ok else "no")

        if resp.source_tables:
            st.write("**Source table(s):**", ", ".join(resp.source_tables))
        if resp.sql:
            st.code(resp.sql)
        if resp.data:
            st.dataframe(resp.data, use_container_width=True)


def _missing_snowflake_creds() -> Optional[str]:
    # Required to create a Snowpark Session for Cortex calls.
    required = ["account", "user", "password"]
    missing = [k for k in required if not (SNOWFLAKE_CONNECTION.get(k) or "").strip()]
    if not missing:
        return None
    return (
        "Snowflake credentials are not configured. "
        "Set `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, and `SNOWFLAKE_PASSWORD` "
        "as environment variables (or Streamlit secrets) to run the app."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="centered")

    st.title(APP_TITLE)
    st.caption(
        "Ask about rent burden by US state, county, or (if needed) census block group. "
        "This UI wraps your existing `PopulationChatbot` without changing its logic."
    )

    with st.sidebar:
        st.header("Settings")
        debug_enabled = st.toggle("Debug logs", value=False)
        _set_debug(debug_enabled)

        if st.button("New conversation", type="secondary", use_container_width=True):
            if "bot" in st.session_state:
                del st.session_state["bot"]
            st.rerun()

        st.divider()
        st.write("**Deployment note:** This app requires Snowflake Cortex access for LLM calls.")

    missing_msg = _missing_snowflake_creds()
    if missing_msg:
        st.error(missing_msg)
        st.stop()

    bot = _get_bot()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a question (e.g., 'What is rent burden in California?')")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = bot.handle_user_question(prompt)
        _render_answer(resp)

    st.session_state["messages"].append({"role": "assistant", "content": resp.user_message})


if __name__ == "__main__":
    # Streamlit runs `main()` on import, but keep CLI-friendly execution too.
    main()

