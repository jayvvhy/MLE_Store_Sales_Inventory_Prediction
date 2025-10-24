import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.ml.feature import VectorAssembler, StandardScaler

# import utils.data_processing_bronze_table
# import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python gold_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
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


    # ---------------- Load Gold features ----------------
    gold_path = "/opt/airflow/datamart/gold/"
    print(f"Loading features from: {gold_path}")
    df = spark.read.parquet("/opt/airflow/datamart/gold/")
    print(f"✅ Rows loaded: {df.count()}")

    target_col = "sales"
    numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    feature_cols = [c for c in numeric_cols if c != target_col]


    # ---------------- Assemble + Scale ----------------
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )
    scaled = scaler.fit(assembled).transform(assembled)

    # ---------------- Train / Test / OOT Split ----------------
    train_test_df = scaled.filter(
        F.col("date").between(F.lit(tt_start), F.lit(tt_end))
    )
    oot_df = scaled.filter(F.col("date").between(F.lit(oot_start), F.lit(oot_end)))

    train_df, test_df = train_test_df.randomSplit(
        [train_test_ratio, 1 - train_test_ratio], seed=88
    )

    print(
        f"Train: {train_df.count()} | Test: {test_df.count()} | OOT: {oot_df.count()}"
    )

    # ---------------- Save to Parquet ----------------
    out_dir = "/opt/airflow/scripts/datamart/train_test/"
    os.makedirs(out_dir, exist_ok=True)
    stamp = snapshotdate.replace("-", "_")

    train_df.select("features", target_col).write.mode("overwrite").parquet(
        os.path.join(out_dir, f"train_{stamp}.parquet")
    )
    test_df.select("features", target_col).write.mode("overwrite").parquet(
        os.path.join(out_dir, f"test_{stamp}.parquet")
    )
    oot_df.select("features", target_col).write.mode("overwrite").parquet(
        os.path.join(out_dir, f"oot_{stamp}.parquet")
    )

    print(f"✅ Saved processed splits to {out_dir}")

    spark.stop()
    print("\n\n--- Completed job ---\n\n")

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)