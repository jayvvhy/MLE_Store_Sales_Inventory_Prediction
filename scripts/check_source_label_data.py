import sys
import argparse
import pandas as pd
import os

def main(snapshotdate: str):
    data_path = "/opt/airflow/scripts/data/train.csv"

    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(data_path)

    # Identify which column contains date info
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if not date_col:
        print("❌ No date-like column found in train.csv")
        sys.exit(1)

    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Check for the snapshot date
    snapshot = pd.to_datetime(snapshotdate)
    rows = df[df[date_col] == snapshot]

    if len(rows) > 0:
        print(f"✅ Found {len(rows)} rows for {snapshotdate}")
    else:
        print(f"❌ No data found for {snapshotdate}")
        sys.exit(1)  # Exit 1 means Airflow will mark task as FAILED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()
    main(args.snapshotdate)