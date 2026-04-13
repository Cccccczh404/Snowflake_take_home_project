# US Population Rent Burden Chatbot (Streamlit)

This repo contains your existing rent-burden chatbot (`chatbot.py`) plus a Streamlit UI wrapper (`streamlit_app.py`).

## Deploy (public URL, no local setup)

Use **Streamlit Community Cloud**:

- **App file**: `streamlit_app.py`
- **Python deps**: `requirements.txt`

### Configure secrets (required)

The chatbot uses **Snowflake Cortex** for LLM calls (data queries run locally from the CSVs).

In Streamlit Cloud, set these secrets (App → Settings → Secrets):

```toml
SNOWFLAKE_ACCOUNT = "..."
SNOWFLAKE_USER = "..."
SNOWFLAKE_PASSWORD = "..."
SNOWFLAKE_AUTHENTICATOR = "snowflake"
SNOWFLAKE_ROLE = "ACCOUNTADMIN"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
SNOWFLAKE_SCHEMA = "PUBLIC"
```

After deployment, Streamlit will provide a public URL for the app.

## What was changed

- Added `streamlit_app.py` to provide a web UI that calls `PopulationChatbot.handle_user_question()` directly.
- Updated `utils/config.py` so Snowflake credentials come from environment variables / secrets (safe for public deployment).

