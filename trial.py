import snowflake.connector
from snowflake.connector import DictCursor

conn = snowflake.connector.connect(
    account="BCUDXAG-LVC55427",
    user="ZHIHENGCHENG",
    password="passwdBilly3737!",   # ⚠️ don't hardcode in real projects
    authenticator="snowflake",
    warehouse="COMPUTE_WH",
    database="US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    schema="PUBLIC"
)

cur = conn.cursor(DictCursor)

# ===== choose your table =====
table_name = '2019_RENT_PERCENTAGE_HOUSEHOLD_INCOME'
quoted = f'"{table_name}"'

print("\n=== DATASET INFO ===")

# 1. Row count
cur.execute(f"SELECT COUNT(*) AS total_rows FROM {quoted}")
row_count = cur.fetchone()["TOTAL_ROWS"]
print(f"Total rows: {row_count}")

# 2. Table size
cur.execute(f"""
SELECT
    row_count,
    bytes,
    ROUND(bytes / 1024 / 1024, 2) AS size_mb
FROM INFORMATION_SCHEMA.TABLES
WHERE table_name = '{table_name}'
""")
size_info = cur.fetchone()
print(f"Size (MB): {size_info['SIZE_MB']}")

# 3. Columns
print("\n=== COLUMNS ===")
cur.execute(f"DESC TABLE {quoted}")
cols = cur.fetchall()
for c in cols:
    print(f"{c['name']} | {c['type']} | nullable={c['null?']}")

# 4. Sample rows
print("\n=== SAMPLE ROWS (5 rows) ===")
cur.execute(f"SELECT * FROM {quoted} LIMIT 5")
rows = cur.fetchall()
for r in rows:
    print(r)

cur.close()
conn.close()