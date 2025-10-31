import argparse
import os
import glob
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.ml.feature import VectorAssembler, StandardScaler


def main(snapshotdate):
    print('\n\n--- Starting Gold Feature Store Job ---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- Config ---
    model_train_date = datetime.strptime(snapshotdate, "%Y-%m-%d").date()
    train_test_period_days = 5
    oot_period_days = 2
    train_test_ratio = 0.8

    oot_end = model_train_date - timedelta(days=1)
    oot_start = model_train_date - relativedelta(days=oot_period_days)
    tt_end = oot_start - timedelta(days=1)
    tt_start = oot_start - relativedelta(days=train_test_period_days)

    config = {
        "model_train_date": model_train_date,
        "train_test_ratio": train_test_ratio,
        "oot_start": oot_start,
        "oot_end": oot_end,
        "train_test_start": tt_start,
        "train_test_end": tt_end,
    }
    pprint.pprint(config)

    # --- Directories ---
    gold_dir = "/opt/airflow/datamart/gold/"
    out_dir = "/opt/airflow/scripts/datamart/train_test/"
    os.makedirs(out_dir, exist_ok=True)

    # --- Get all gold parquet files ---
    gold_files = glob.glob(os.path.join(gold_dir, "gold_store*.parquet"))
    if not gold_files:
        raise FileNotFoundError(f"No gold files found in {gold_dir}")

    print(f"📂 Found {len(gold_files)} gold parquet files.")
    
    # --- Loop through each gold file (store-family combination) ---
    for gold_file in gold_files:
        base = os.path.basename(gold_file)
        print(f"\n🚀 Processing {base}")
        df = spark.read.parquet(gold_file)

        if df.count() == 0:
            print(f"⚠️ Skipping {base} (empty file)")
            continue

        # Identify target variable
        target_col = "weekly_sales_sum"
        numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
        feature_cols = [c for c in numeric_cols if c != target_col]

        # Fill missing values
        fill_defaults = {c: 0.0 for c in feature_cols}
        df = df.fillna(fill_defaults)

        # --- Assemble & Scale ---
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        assembled = assembler.transform(df)

        scaler = StandardScaler(
            inputCol="features_raw", outputCol="features", withMean=True, withStd=True
        )
        scaled = scaler.fit(assembled).transform(assembled)

        # --- Train/Test/OOT Split ---
        train_test_df = scaled.filter(F.col("week_start").between(F.lit(tt_start), F.lit(tt_end)))
        oot_df = scaled.filter(F.col("week_start").between(F.lit(oot_start), F.lit(oot_end)))

        train_df, test_df = train_test_df.randomSplit([train_test_ratio, 1 - train_test_ratio], seed=88)

        print(f"✅ Train: {train_df.count()} | Test: {test_df.count()} | OOT: {oot_df.count()}")

        # --- Save Splits ---
        combo_name = base.replace(".parquet", "").replace("gold_", "")
        train_df.select("features", target_col).write.mode("overwrite").parquet(
            os.path.join(out_dir, f"train_{combo_name}_{snapshotdate.replace('-', '_')}.parquet")
        )
        test_df.select("features", target_col).write.mode("overwrite").parquet(
            os.path.join(out_dir, f"test_{combo_name}_{snapshotdate.replace('-', '_')}.parquet")
        )
        oot_df.select("features", target_col).write.mode("overwrite").parquet(
            os.path.join(out_dir, f"oot_{combo_name}_{snapshotdate.replace('-', '_')}.parquet")
        )

        print(f"💾 Saved splits for {combo_name} to {out_dir}")

    print("\n🎉 Completed gold feature processing for all combinations!\n")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gold feature store")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
