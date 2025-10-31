import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType


import os
import glob
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def process_labels_gold_table(snapshot_date_str, silver_directory, gold_label_store_directory, spark):
    # --- Prepare arguments ---
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_tag = snapshot_date_str.replace("-", "_")

    # --- Load all silver parquet files for that snapshot date ---
    silver_pattern = os.path.join(silver_directory, f"silver_store*_{date_tag}.parquet")
    silver_files = glob.glob(silver_pattern)

    if not silver_files:
        raise FileNotFoundError(f"❌ No silver store files found for {snapshot_date_str} in {silver_directory}")

    print(f"📦 Found {len(silver_files)} silver files for {snapshot_date_str}")
    for f in silver_files:
        print(f"  - {os.path.basename(f)}")

    # --- Load holiday file ---
    holiday_path = os.path.join(silver_directory, f"silver_holiday_{date_tag}.parquet")
    df_h = spark.read.parquet(holiday_path)
    df_h = df_h.filter(F.col("locale") == "National")
    print(f"✅ Loaded holiday data: {df_h.count()} rows")

    os.makedirs(gold_label_store_directory, exist_ok=True)

    # --- Loop through each silver file (per store/family) ---
    for silver_file in silver_files:
        base = os.path.basename(silver_file)
        print(f"\n🚀 Processing {base}")

        df = spark.read.parquet(silver_file)
        count = df.count()
        if count == 0:
            print(f"⚠️ Skipping {base} (0 rows)")
            continue

        # Extract store/family from filename
        parts = base.replace(".parquet", "").split("_")
        store_id = [p for p in parts if p.startswith("store")][0].replace("store", "")
        # family name comes *right after* the store part (index after "storeX")
        store_index = parts.index(f"store{store_id}")
        family_name = parts[store_index + 1] if len(parts) > store_index + 1 else "unknown"
        family_name_clean = family_name.lower().replace("/", "").replace(" ", "")

        # --- Join holiday info ---
        df_gold = (
            df.join(df_h.select("date", "locale"), on="date", how="left")
              .withColumn("is_holiday", F.when(F.col("locale").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
              .drop("locale")
        )

        # Add week_start (Monday) and day_of_week
        df_gold = df_gold.withColumn("week_start", F.date_trunc("week", F.col("date")))

        # Aggregate by week for each store/family combination
        df_weekly = (
            df_gold.groupBy("store_nbr", "family", "week_start")
                .agg(
                    F.sum("sales").alias("weekly_sales_sum"),
                    F.mean("sales").alias("weekly_sales_avg"),
                    F.countDistinct("date").alias("days_in_week"),
                    F.max("is_holiday").alias("any_holiday")
                )
                .orderBy("store_nbr", "family", "week_start")
        )

        # --- Rolling features (7d, 14d, 28d) ---
        window_spec = (
            Window.partitionBy("store_nbr", "family")
                  .orderBy("week_start")
                  .rowsBetween(-3, -1)
        )

        df_weekly = (
            df_weekly
            .withColumn("avg_sales_14d", F.avg("weekly_sales_avg")
                        .over(Window.partitionBy("store_nbr", "family")
                        .orderBy("week_start")
                        .rowsBetween(-2, -1)))
            .withColumn("avg_sales_28d", F.avg("weekly_sales_avg").over(window_spec))
            .withColumn("sum_sales_14d", F.sum("weekly_sales_sum")
                        .over(Window.partitionBy("store_nbr", "family")
                        .orderBy("week_start")
                        .rowsBetween(-2, -1)))
            .withColumn("sum_sales_28d", F.sum("weekly_sales_sum").over(window_spec))
        )

        df_weekly = df_weekly.fillna({
            "avg_sales_14d": 0.0,
            "avg_sales_28d": 0.0,
            "sum_sales_14d": 0.0,
            "sum_sales_28d": 0.0
        })

        # --- Save per-combination gold file ---
        gold_filename = f"gold_store{store_id}_{family_name_clean}_{date_tag}.parquet"
        gold_path = os.path.join(gold_label_store_directory, gold_filename)

        df_weekly.write.mode("overwrite").parquet(gold_path)
        print(f"✅ Saved {df_weekly.count()} rows to {gold_path}")

    print("\n🎉 Completed gold processing for all store/family combinations!\n")

