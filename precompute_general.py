import os
import sys
import time
from typing import Dict, List

import pandas as pd
import requests
import snowflake.connector

# ============================================================
# CONFIG
# ============================================================

SNOWFLAKE_CONFIG: Dict[str, str] = {
    "account": "YOUR_ACCOUNT",
    "user": "YOUR_USER",
    "password": "YOUR_PASSWORD",
    "authenticator": "snowflake",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    "schema": "PUBLIC",
}

SOURCE_DB = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
SOURCE_SCHEMA = "PUBLIC"
SOURCE_TABLE = "2019_RENT_PERCENTAGE_HOUSEHOLD_INCOME"

ACS_YEAR = "2019"
ACS_DATASET = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

REQUEST_TIMEOUT = 60
REQUEST_SLEEP_SEC = 0.15

OUTPUT_DIR = r".\output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cbg_population_features.csv")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "cbg_population_features.parquet")


# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def api_get_json(url: str, params: Dict[str, str]):
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_json_to_df(rows) -> pd.DataFrame:
    if not rows or len(rows) < 2:
        return pd.DataFrame()
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_source_table(conn) -> pd.DataFrame:
    sql = f'''
        SELECT *
        FROM "{SOURCE_DB}"."{SOURCE_SCHEMA}"."{SOURCE_TABLE}"
    '''
    log("Reading source dataset from Snowflake...")
    return pd.read_sql(sql, conn)


def derive_cbg_codes(df: pd.DataFrame) -> pd.DataFrame:
    log("Deriving CBG geographic codes...")
    out = df.copy()

    out["CBG_FIPS"] = (
        out["CENSUS_BLOCK_GROUP"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(12)
    )

    out["STATE_FIPS"] = out["CBG_FIPS"].str.slice(0, 2)
    out["COUNTY_FIPS"] = out["CBG_FIPS"].str.slice(0, 5)
    out["TRACT_FIPS"] = out["CBG_FIPS"].str.slice(0, 11)
    out["BLOCK_GROUP"] = out["CBG_FIPS"].str.slice(11, 12)

    out["COUNTY_CODE_WITHIN_STATE"] = out["CBG_FIPS"].str.slice(2, 5)
    out["TRACT_CODE_WITHIN_COUNTY"] = out["CBG_FIPS"].str.slice(5, 11)

    return out


def fetch_states_lookup(needed_state_fips: List[str]) -> pd.DataFrame:
    log("Fetching state names from Census API...")
    rows = api_get_json(
        ACS_DATASET,
        {
            "get": "NAME",
            "for": "state:*",
        },
    )
    df = api_json_to_df(rows)
    df = df.rename(columns={"state": "STATE_FIPS", "NAME": "STATE_NAME"})
    df["STATE_FIPS"] = df["STATE_FIPS"].str.zfill(2)
    df = df[df["STATE_FIPS"].isin(needed_state_fips)].copy()
    return df[["STATE_FIPS", "STATE_NAME"]].drop_duplicates()


def fetch_counties_lookup(needed_state_fips: List[str]) -> pd.DataFrame:
    log("Fetching county names from Census API...")
    pieces = []

    for state_fips in sorted(set(needed_state_fips)):
        rows = api_get_json(
            ACS_DATASET,
            {
                "get": "NAME",
                "for": "county:*",
                "in": f"state:{state_fips}",
            },
        )
        df = api_json_to_df(rows)
        df = df.rename(
            columns={
                "state": "STATE_FIPS",
                "county": "COUNTY_CODE",
                "NAME": "COUNTY_NAME",
            }
        )
        df["STATE_FIPS"] = df["STATE_FIPS"].str.zfill(2)
        df["COUNTY_CODE"] = df["COUNTY_CODE"].str.zfill(3)
        df["COUNTY_FIPS"] = df["STATE_FIPS"] + df["COUNTY_CODE"]
        pieces.append(df[["STATE_FIPS", "COUNTY_CODE", "COUNTY_FIPS", "COUNTY_NAME"]].drop_duplicates())
        time.sleep(REQUEST_SLEEP_SEC)

    if not pieces:
        return pd.DataFrame(columns=["STATE_FIPS", "COUNTY_CODE", "COUNTY_FIPS", "COUNTY_NAME"])

    return pd.concat(pieces, ignore_index=True).drop_duplicates()


def enrich_with_names(df: pd.DataFrame) -> pd.DataFrame:
    needed_states = sorted(df["STATE_FIPS"].dropna().astype(str).unique().tolist())

    states = fetch_states_lookup(needed_states)
    counties = fetch_counties_lookup(needed_states)

    log("Joining readable geography names...")
    out = df.merge(states, on="STATE_FIPS", how="left")
    out = out.merge(
        counties[["COUNTY_FIPS", "COUNTY_NAME"]],
        on="COUNTY_FIPS",
        how="left",
    )

    out["BLOCK_GROUP_NAME"] = "Block Group " + out["BLOCK_GROUP"].fillna("")
    out["COUNTY_STATE_NAME"] = (
        out["COUNTY_NAME"].fillna("Unknown County")
        + ", "
        + out["STATE_NAME"].fillna("Unknown State")
    )
    out["FULL_GEO_NAME"] = (
        out["BLOCK_GROUP_NAME"].fillna("")
        + ", "
        + out["COUNTY_NAME"].fillna("Unknown County")
        + ", "
        + out["STATE_NAME"].fillna("Unknown State")
    )

    return out


def save_local(df: pd.DataFrame) -> None:
    log(f"Saving locally to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV to: {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Saved Parquet to: {OUTPUT_PARQUET}")
    except Exception as e:
        print(f"Parquet save skipped: {e}")


def show_summary(df: pd.DataFrame) -> None:
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    for c in df.columns:
        print("-", c)

    print("\nSample enriched rows:")
    cols = [
        "CENSUS_BLOCK_GROUP",
        "CBG_FIPS",
        "STATE_FIPS",
        "STATE_NAME",
        "COUNTY_FIPS",
        "COUNTY_NAME",
        "BLOCK_GROUP",
        "FULL_GEO_NAME",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].head(5).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    conn = None
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)

        source_df = fetch_source_table(conn)
        enriched_df = derive_cbg_codes(source_df)
        enriched_df = enrich_with_names(enriched_df)

        save_local(enriched_df)
        show_summary(enriched_df)

        print("\nDone.")

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()