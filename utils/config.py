from __future__ import annotations

import os

try:
    # Allows local dev via a .env file without hardcoding secrets.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # If python-dotenv isn't installed (or .env is absent), fall back to real env vars.
    pass

# ============================================================
# LOCAL DATA FILES  (relative to the project root)
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_CSV_FILES = {
    "STATE_SUMMARY":           os.path.join(_PROJECT_ROOT, "output", "state_summary.csv"),
    "COUNTY_SUMMARY":          os.path.join(_PROJECT_ROOT, "output", "county_summary.csv"),
    "CBG_POPULATION_FEATURES": os.path.join(_PROJECT_ROOT, "output", "cbg_population_features.csv"),
}

# ============================================================
# SNOWFLAKE CONNECTION  (used only for Cortex LLM calls)
# ============================================================

SNOWFLAKE_CONNECTION = {
    # NOTE: For security, never hardcode credentials in code.
    # Provide these via environment variables or Streamlit secrets.
    "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
    "user": os.getenv("SNOWFLAKE_USER", ""),
    "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
    "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
    "role": os.getenv("SNOWFLAKE_ROLE", ""),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", ""),
    "database": os.getenv("SNOWFLAKE_DATABASE", ""),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", ""),
}

# ============================================================
# CORTEX LLM
# ============================================================

CORTEX_MODEL = "mistral-large2"

# ============================================================
# TABLE CATALOG
# Each entry describes a Snowflake table the chatbot can query.
# Used by tool_pick_tables and tool_generate_sql.
# ============================================================

TABLE_CATALOG: dict = {
    "STATE_SUMMARY": {
        "description": (
            "Pre-aggregated rent burden statistics at the US state level. "
            "Use this for state comparisons, national rankings, or any question "
            "about a specific state's overall rent burden."
        ),
        "grain": "state",
        "use_when": (
            "question asks about a specific state, compares multiple states, "
            "asks for national state-level rankings, or does not require county/neighborhood detail"
        ),
        "notes": (
            "No column names require double-quoting. "
            "Filter on STATE_NAME using ILIKE, e.g. WHERE STATE_NAME ILIKE '%california%'. "
            "Ratio columns are decimals 0.0–1.0; multiply by 100 to express as a percentage."
        ),
        "sample_values": {
            "STATE_FIPS": "6, 48, 36, 12",
            "STATE_NAME": "'California', 'Texas', 'New York', 'Florida'",
            "total_renters": "5889686, 2500000",
            "rent_burden_ratio_30_plus": "0.52, 0.45  (decimal fraction, NOT percent)",
        },
        "columns": {
            "STATE_FIPS":                  ("INTEGER", "Numeric FIPS code for the state, e.g. 6 for California"),
            "STATE_NAME":                  ("VARCHAR",  "Full state name, e.g. 'California'. Use ILIKE for filtering."),
            "cbg_count":                   ("INTEGER", "Number of census block groups in the state"),
            "total_renters":               ("BIGINT",  "Total renter-occupied housing units in the state"),
            "burden_30_plus":              ("BIGINT",  "Count of renters paying 30% or more of income on rent"),
            "burden_50_plus":              ("BIGINT",  "Count of renters paying 50% or more of income on rent (severely burdened)"),
            "burden_under_20":             ("BIGINT",  "Count of renters paying less than 20% of income on rent"),
            "not_computed":                ("BIGINT",  "Count of renters where rent burden could not be computed"),
            "rent_burden_ratio_30_plus":   ("FLOAT",   "Fraction of renters paying 30%+ (0.0 to 1.0). Multiply by 100 for percent."),
            "severe_burden_ratio_50_plus": ("FLOAT",   "Fraction of renters paying 50%+ (0.0 to 1.0). Multiply by 100 for percent."),
            "low_burden_ratio_under_20":   ("FLOAT",   "Fraction of renters paying under 20% (0.0 to 1.0). Multiply by 100 for percent."),
        },
    },
    "COUNTY_SUMMARY": {
        "description": (
            "Pre-aggregated rent burden statistics at the US county level. "
            "Use this for county comparisons, within-state county rankings, or any question "
            "about a specific county's rent burden."
        ),
        "grain": "county",
        "use_when": (
            "question asks about a specific county, compares counties, "
            "asks for county-level rankings within a state, or mentions 'county'"
        ),
        "notes": (
            "No column names require double-quoting. "
            "COUNTY_NAME already contains the state name, e.g. 'Los Angeles County, California'. "
            "To filter by county use ILIKE on COUNTY_NAME. "
            "To filter by state use ILIKE on STATE_NAME. "
            "Ratio columns are decimals 0.0–1.0."
        ),
        "sample_values": {
            "STATE_FIPS":   "6, 48",
            "STATE_NAME":   "'California', 'Texas'",
            "COUNTY_FIPS":  "6037, 48201",
            "COUNTY_NAME":  "'Los Angeles County, California', 'Harris County, Texas'",
            "county_state_name": "'Los Angeles County, California, California'",
            "total_renters": "1797279, 650000",
            "rent_burden_ratio_30_plus": "0.548, 0.42  (decimal fraction)",
        },
        "columns": {
            "STATE_FIPS":                  ("INTEGER", "Numeric FIPS code for the state"),
            "STATE_NAME":                  ("VARCHAR",  "Full state name, e.g. 'California'. Use ILIKE for filtering."),
            "COUNTY_FIPS":                 ("INTEGER", "Numeric FIPS code for the county, e.g. 6037"),
            "COUNTY_NAME":                 ("VARCHAR",  "County + state, e.g. 'Los Angeles County, California'. Use ILIKE for filtering."),
            "county_state_name":           ("VARCHAR",  "Alternate readable county+state label"),
            "cbg_count":                   ("INTEGER", "Number of census block groups in the county"),
            "total_renters":               ("BIGINT",  "Total renter-occupied housing units in the county"),
            "burden_30_plus":              ("BIGINT",  "Count of renters paying 30%+ of income on rent"),
            "burden_50_plus":              ("BIGINT",  "Count of renters paying 50%+ of income on rent"),
            "burden_under_20":             ("BIGINT",  "Count of renters paying less than 20% of income on rent"),
            "not_computed":                ("BIGINT",  "Count of renters where burden was not computed"),
            "rent_burden_ratio_30_plus":   ("FLOAT",   "Fraction of renters paying 30%+ (0.0 to 1.0)"),
            "severe_burden_ratio_50_plus": ("FLOAT",   "Fraction of renters paying 50%+ (0.0 to 1.0)"),
            "low_burden_ratio_under_20":   ("FLOAT",   "Fraction of renters paying under 20% (0.0 to 1.0)"),
        },
    },
    "CBG_POPULATION_FEATURES": {
        "description": (
            "Raw census block group (CBG) level rent burden data — most granular dataset. "
            "Use only when the question explicitly asks about a neighborhood, tract, or block group."
        ),
        "grain": "census_block_group",
        "use_when": (
            "question asks about a specific neighborhood, census block group, tract, "
            "or needs finer granularity than county"
        ),
        "notes": (
            "CRITICAL: All column names that contain colons, dots, or spaces MUST be "
            "wrapped in double quotes in SQL, e.g. \"Total: Renter-occupied housing units\". "
            "CENSUS_BLOCK_GROUP is a 12-digit string FIPS code — do NOT cast to INTEGER. "
            "Filter by STATE_NAME or COUNTY_NAME using ILIKE."
        ),
        "sample_values": {
            "CENSUS_BLOCK_GROUP": "'421010137003'  (12-digit string)",
            "STATE_NAME":  "'Pennsylvania', 'California'",
            "COUNTY_NAME": "'Philadelphia County, Pennsylvania'",
            "FULL_GEO_NAME": "'Block Group 3, Philadelphia County, Pennsylvania'",
        },
        "columns": {
            "CENSUS_BLOCK_GROUP":   ("VARCHAR", "12-digit FIPS code for the census block group. MUST be treated as string."),
            "CBG_FIPS":             ("VARCHAR", "Same as CENSUS_BLOCK_GROUP"),
            "STATE_FIPS":           ("INTEGER", "State FIPS code"),
            "COUNTY_FIPS":          ("INTEGER", "County FIPS code"),
            "TRACT_FIPS":           ("INTEGER", "Census tract FIPS code"),
            "STATE_NAME":           ("VARCHAR", "State name, e.g. 'Pennsylvania'"),
            "COUNTY_NAME":          ("VARCHAR", "County name including state, e.g. 'Philadelphia County, Pennsylvania'"),
            "FULL_GEO_NAME":        ("VARCHAR", "Full geographic label including block group, county, and state"),
            '"Total: Renter-occupied housing units"':          ("INTEGER", "Total renters in the block group. MUST be double-quoted in SQL."),
            '"50.0 percent or more: Renter-occupied housing units"': ("INTEGER", "Severely burdened renters. MUST be double-quoted in SQL."),
            '"40.0 to 49.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 40–49.9%. MUST be double-quoted in SQL."),
            '"35.0 to 39.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 35–39.9%. MUST be double-quoted in SQL."),
            '"30.0 to 34.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 30–34.9%. MUST be double-quoted in SQL."),
            '"25.0 to 29.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 25–29.9%. MUST be double-quoted in SQL."),
            '"20.0 to 24.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 20–24.9%. MUST be double-quoted in SQL."),
            '"15.0 to 19.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 15–19.9%. MUST be double-quoted in SQL."),
            '"10.0 to 14.9 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying 10–14.9%. MUST be double-quoted in SQL."),
            '"Less than 10.0 percent: Renter-occupied housing units"': ("INTEGER", "Renters paying less than 10%. MUST be double-quoted in SQL."),
            '"Not computed: Renter-occupied housing units"':    ("INTEGER", "Renters where burden not computed. MUST be double-quoted in SQL."),
        },
    },
}

# Appended to all Cortex system prompts (agents + SQL tools) for consistent geography semantics.
LLM_POPULATION_GUIDANCE = (
    "Questions about population for a specific area are on-topic: answer with SUM of "
    "renter-occupied housing unit counts for that geography (use total_renters on "
    "STATE_SUMMARY or COUNTY_SUMMARY, or SUM of \"Total: Renter-occupied housing units\" on "
    "CBG_POPULATION_FEATURES for neighborhood / block-group questions). "
    "Treat \"us\", \"US\", \"nationwide\", \"the country\", and similar fuzzy phrases as the "
    "United States and aggregate national totals by summing those counts across all rows in STATE_SUMMARY."
)

# ============================================================
# QUERY LIMITS
# ============================================================

MAX_RESULT_ROWS = 100
MAX_PREVIEW_ROWS = 10

# ============================================================
# FORBIDDEN SQL KEYWORDS (for validation)
# ============================================================

FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "MERGE", "GRANT", "REVOKE", "EXECUTE", "EXEC",
    "CALL", "BEGIN", "COMMIT", "ROLLBACK",
]
