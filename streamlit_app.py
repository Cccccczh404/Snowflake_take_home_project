from __future__ import annotations

import os

import streamlit as st

from chatbot import PopulationChatbot
from utils import SNOWFLAKE_CONNECTION, SnowflakeAgentClient


def _get_snowpark_session():
    """
    Prefer Streamlit in Snowflake's active session when available.
    Fall back to creating a Snowpark Session from Streamlit secrets (Streamlit Cloud)
    or environment variables.
    """
    try:
        from snowflake.snowpark.context import get_active_session  # type: ignore

        return get_active_session()
    except Exception:
        pass

    try:
        import snowflake.snowpark as snowpark

        if "snowflake" in st.secrets:
            return snowpark.Session.builder.configs(dict(st.secrets["snowflake"])).create()
        if "connections" in st.secrets and "snowflake" in st.secrets["connections"]:
            return snowpark.Session.builder.configs(dict(st.secrets["connections"]["snowflake"])).create()
    except Exception:
        # We'll fail below with a clearer message.
        return None

    # Last-chance fallback: environment variables via utils/config.py
    if any(v for v in SNOWFLAKE_CONNECTION.values()):
        import snowflake.snowpark as snowpark

        return snowpark.Session.builder.configs(SNOWFLAKE_CONNECTION).create()

    return None


@st.cache_resource(show_spinner="Initializing Snowflake + loading local data…")
def _get_client_and_bot() -> tuple[SnowflakeAgentClient, PopulationChatbot]:
    session = _get_snowpark_session()
    if session is None:
        raise RuntimeError(
            "Missing Snowflake connection. Configure Streamlit secrets under key "
            "`snowflake` (or `connections.snowflake`), or set SNOWFLAKE_* env vars."
        )
    client = SnowflakeAgentClient(session=session)
    bot = PopulationChatbot(client)
    return client, bot


def main() -> None:
    st.set_page_config(page_title="US Rent Burden Chatbot", page_icon="🏠", layout="centered")

    st.title("US Population Rent Burden Chatbot")
    st.caption("Ask about rent burden by state or county (data is queried locally; Cortex is used for LLM steps).")

    with st.sidebar:
        st.subheader("Connection")
        st.write(
            "This app uses a Snowflake session for Cortex LLM calls. "
            "In Streamlit in Snowflake it uses your active session automatically."
        )
        st.subheader("Controls")
        if st.button("New conversation", use_container_width=True):
            st.session_state.pop("conversation", None)
            st.session_state.pop("messages", None)
            st.rerun()

    client, bot = _get_client_and_bot()

    if "conversation" not in st.session_state:
        st.session_state.conversation = bot
    else:
        # Keep cached client, but preserve the per-user conversation state.
        bot = st.session_state.conversation

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("extras"):
                with st.expander("Details", expanded=False):
                    if m["extras"].get("status"):
                        st.code(m["extras"]["status"])
                    if m["extras"].get("sql"):
                        st.code(m["extras"]["sql"])
                    if m["extras"].get("source_tables"):
                        st.write("**Source tables:**", ", ".join(m["extras"]["source_tables"]))
                    if m["extras"].get("row_count") is not None:
                        st.write("**Rows returned:**", m["extras"]["row_count"])

    prompt = st.chat_input("Ask a question (e.g., “Which state has the highest severe rent burden?”)")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = bot.handle_user_question(prompt)
        st.markdown(resp.user_message)

        extras = {
            "status": f"{resp.status} ({resp.route})",
            "sql": resp.sql,
            "source_tables": resp.source_tables,
            "row_count": resp.row_count,
        }
        st.session_state.messages.append(
            {"role": "assistant", "content": resp.user_message, "extras": extras}
        )

    # Keep connection alive for subsequent turns
    _ = client


if __name__ == "__main__":
    # Allow Streamlit to run this file directly.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

