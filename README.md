# US Population Rent Burden Chatbot (Web UI)

This repo contains a rent-burden chatbot:
- Data queries run **locally** against `output/*.csv` (loaded into an in-memory SQLite DB).
- LLM steps (intent routing, SQL generation/validation, grounded answer, tone synthesis) call **Snowflake Cortex** via Snowpark.

## Web interface (Streamlit)

The web UI lives in `streamlit_app.py`.

### Option A (recommended): Streamlit in Snowflake

Snowflake’s docs describe how Streamlit in Snowflake apps can be shared via **app-viewer URLs** (a clean URL without Snowsight chrome) and **app-builder URLs**:
- Snowflake docs: [Sharing Streamlit in Snowflake apps](https://docs.snowflake.com/developer-guide/streamlit/features/sharing-streamlit-apps)

At a high level:
1. Create a Streamlit app in Snowflake (Snowsight → Projects → Streamlit) and upload `streamlit_app.py` as the app code.
2. Ensure the role/user running the app can use Cortex and has access to required objects.
3. Use the Streamlit app **Share** feature to copy an **app-viewer URL** for evaluation.

Snowflake also supports managing apps via the Snowflake CLI (including retrieving the app URL):
- Snowflake docs: [Managing Streamlit apps](https://docs.snowflake.com/en/developer-guide/snowflake-cli/streamlit-apps/manage-apps/manage-app)

### Option B: Streamlit Community Cloud (public internet)

If you want a publicly accessible URL (no Snowflake login), deploy this repo on Streamlit Community Cloud.

You must provide Snowflake credentials as Streamlit secrets (never hardcode them).

In Streamlit Cloud, set secrets either as:

```toml
# .streamlit/secrets.toml (Streamlit Cloud secrets UI)
[snowflake]
account = "..."
user = "..."
password = "..."
role = "..."
warehouse = "..."
database = "..."
schema = "..."
authenticator = "snowflake"
```

or:

```toml
[connections.snowflake]
account = "..."
user = "..."
password = "..."
role = "..."
warehouse = "..."
database = "..."
schema = "..."
authenticator = "snowflake"
```

Then set the app entrypoint to `streamlit_app.py`.

## Security

`utils/config.py` reads Snowflake connection settings from environment variables (`SNOWFLAKE_*`) so secrets aren’t committed.

# Snowflake_take_home_project
