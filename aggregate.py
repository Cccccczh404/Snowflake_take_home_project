import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = r".\output\cbg_population_features.csv"
OUTPUT_DIR = r".\output"

STATE_CSV = os.path.join(OUTPUT_DIR, "state_summary.csv")
STATE_PARQUET = os.path.join(OUTPUT_DIR, "state_summary.parquet")

COUNTY_CSV = os.path.join(OUTPUT_DIR, "county_summary.csv")
COUNTY_PARQUET = os.path.join(OUTPUT_DIR, "county_summary.parquet")


# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def safe_divide(a, b):
    return a / b if b not in [0, None] else None


# ============================================================
# LOAD
# ============================================================

log("Loading enriched dataset...")
df = pd.read_csv(INPUT_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_cols = [
    "Total: Renter-occupied housing units",
    "50.0 percent or more: Renter-occupied housing units",
    "Not computed: Renter-occupied housing units",
    "Less than 10.0 percent: Renter-occupied housing units",
    "10.0 to 14.9 percent: Renter-occupied housing units",
    "15.0 to 19.9 percent: Renter-occupied housing units",
    "20.0 to 24.9 percent: Renter-occupied housing units",
    "25.0 to 29.9 percent: Renter-occupied housing units",
    "30.0 to 34.9 percent: Renter-occupied housing units",
    "35.0 to 39.9 percent: Renter-occupied housing units",
    "40.0 to 49.9 percent: Renter-occupied housing units",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ============================================================
# PRECOMPUTED FEATURES
# ============================================================

log("Computing burden features...")

df["rent_burden_30_plus"] = (
    df["30.0 to 34.9 percent: Renter-occupied housing units"]
    + df["35.0 to 39.9 percent: Renter-occupied housing units"]
    + df["40.0 to 49.9 percent: Renter-occupied housing units"]
    + df["50.0 percent or more: Renter-occupied housing units"]
)

df["rent_burden_50_plus"] = df["50.0 percent or more: Renter-occupied housing units"]

df["rent_burden_under_20"] = (
    df["Less than 10.0 percent: Renter-occupied housing units"]
    + df["10.0 to 14.9 percent: Renter-occupied housing units"]
    + df["15.0 to 19.9 percent: Renter-occupied housing units"]
)

# ============================================================
# STATE SUMMARY
# ============================================================

log("Building state summary...")

state_group_cols = ["STATE_FIPS", "STATE_NAME"]

state_summary = (
    df.groupby(state_group_cols, dropna=False)
    .agg(
        cbg_count=("CENSUS_BLOCK_GROUP", "count"),
        total_renters=("Total: Renter-occupied housing units", "sum"),
        burden_30_plus=("rent_burden_30_plus", "sum"),
        burden_50_plus=("rent_burden_50_plus", "sum"),
        burden_under_20=("rent_burden_under_20", "sum"),
        not_computed=("Not computed: Renter-occupied housing units", "sum"),
    )
    .reset_index()
)

state_summary["rent_burden_ratio_30_plus"] = state_summary.apply(
    lambda r: safe_divide(r["burden_30_plus"], r["total_renters"]), axis=1
)

state_summary["severe_burden_ratio_50_plus"] = state_summary.apply(
    lambda r: safe_divide(r["burden_50_plus"], r["total_renters"]), axis=1
)

state_summary["low_burden_ratio_under_20"] = state_summary.apply(
    lambda r: safe_divide(r["burden_under_20"], r["total_renters"]), axis=1
)

state_summary = state_summary.sort_values(
    by=["total_renters", "STATE_NAME"], ascending=[False, True]
).reset_index(drop=True)

# ============================================================
# COUNTY SUMMARY
# ============================================================

log("Building county summary...")

county_group_cols = ["STATE_FIPS", "STATE_NAME", "COUNTY_FIPS", "COUNTY_NAME"]

county_summary = (
    df.groupby(county_group_cols, dropna=False)
    .agg(
        cbg_count=("CENSUS_BLOCK_GROUP", "count"),
        total_renters=("Total: Renter-occupied housing units", "sum"),
        burden_30_plus=("rent_burden_30_plus", "sum"),
        burden_50_plus=("rent_burden_50_plus", "sum"),
        burden_under_20=("rent_burden_under_20", "sum"),
        not_computed=("Not computed: Renter-occupied housing units", "sum"),
    )
    .reset_index()
)

county_summary["rent_burden_ratio_30_plus"] = county_summary.apply(
    lambda r: safe_divide(r["burden_30_plus"], r["total_renters"]), axis=1
)

county_summary["severe_burden_ratio_50_plus"] = county_summary.apply(
    lambda r: safe_divide(r["burden_50_plus"], r["total_renters"]), axis=1
)

county_summary["low_burden_ratio_under_20"] = county_summary.apply(
    lambda r: safe_divide(r["burden_under_20"], r["total_renters"]), axis=1
)

county_summary["county_state_name"] = (
    county_summary["COUNTY_NAME"].fillna("Unknown County")
    + ", "
    + county_summary["STATE_NAME"].fillna("Unknown State")
)

county_summary = county_summary.sort_values(
    by=["total_renters", "STATE_NAME", "COUNTY_NAME"],
    ascending=[False, True, True]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

log("Saving summary tables...")

state_summary.to_csv(STATE_CSV, index=False)
county_summary.to_csv(COUNTY_CSV, index=False)

print(f"Saved state summary CSV: {STATE_CSV}")
print(f"Saved county summary CSV: {COUNTY_CSV}")

try:
    state_summary.to_parquet(STATE_PARQUET, index=False)
    print(f"Saved state summary Parquet: {STATE_PARQUET}")
except Exception as e:
    print(f"State parquet skipped: {e}")

try:
    county_summary.to_parquet(COUNTY_PARQUET, index=False)
    print(f"Saved county summary Parquet: {COUNTY_PARQUET}")
except Exception as e:
    print(f"County parquet skipped: {e}")

# ============================================================
# PREVIEW
# ============================================================

print("\n=== STATE SUMMARY SAMPLE ===")
print(state_summary.head(10).to_string(index=False))

print("\n=== COUNTY SUMMARY SAMPLE ===")
print(county_summary.head(10).to_string(index=False))

print("\nDone.")