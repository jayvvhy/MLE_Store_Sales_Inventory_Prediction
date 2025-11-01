import os
import sys
import glob
import shutil
import yaml
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, IntegerType, FloatType

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Helper: resolve local vs Airflow paths
# -------------------------------------------------------------------------
def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    base_dir = "/opt/airflow" if os.path.exists("/opt/airflow") else os.getcwd()
    return os.path.join(base_dir, path_str.strip("/"))

# -------------------------------------------------------------------------
# Helper: month formatting
# -------------------------------------------------------------------------
def fmt_tag(dt):
    return dt.strftime("%Y_%m_%d")

# -------------------------------------------------------------------------
# Main process
# -------------------------------------------------------------------------
def process_gold_tables(snapshot_date: str, spark, config_path: str = "config/gold_config.yaml"):
    """
    Build Gold-layer feature_store and label_store for the given snapshot_date.

    Feature store (S): aggregates from current month.
    Label store (S-1): uses next-month (S) total sales as target.
    """

    # ---------------------------------------------------------------------
    # Load config
    # ---------------------------------------------------------------------
    resolved_path = resolve_path(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    silver_dir = resolve_path(config["directories"]["silver_dir"])
    gold_feature_dir = resolve_path(config["directories"]["feature_store"])
    gold_label_dir = resolve_path(config["directories"]["label_store"])
    os.makedirs(gold_feature_dir, exist_ok=True)
    os.makedirs(gold_label_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Prepare snapshot months
    # ---------------------------------------------------------------------
    S = datetime.strptime(snapshot_date, "%Y-%m-%d").date()   # current month for features
    S1 = S - relativedelta(months=1)                           # one month prior (label)
    S2 = S - relativedelta(months=2)
    need = [fmt_tag(S2), fmt_tag(S1), fmt_tag(S)]              # require S, S-1, S-2

    logger.info(f"🚀 Starting Gold processing for snapshot_date={snapshot_date}")

    # ---------------------------------------------------------------------
    # Early sufficiency check (before loading)
    # ---------------------------------------------------------------------
    def _exists(table, tag):
        return os.path.exists(os.path.join(silver_dir, table, f"{tag}.parquet"))

    if not all(_exists("daily_sales", t) for t in need):
        logger.warning(f"⏭️ Missing required Silver files for {need}. Skipping {snapshot_date}.")
        return

    # ---------------------------------------------------------------------
    # Helper to load and union S, S-1, S-2
    # ---------------------------------------------------------------------
    def load_union(table):
        dfs = []
        for t in need:
            path = os.path.join(silver_dir, table, f"{t}.parquet")
            if os.path.exists(path):
                dfs.append(spark.read.parquet(path))
        if not dfs:
            return None
        df_union = dfs[0]
        for d in dfs[1:]:
            df_union = df_union.unionByName(d)
        return df_union

    sales_all = load_union("daily_sales")
    txn_all   = load_union("daily_transactions")
    oil_all   = load_union("oil_prices")

    # ---------------------------------------------------------------------
    # Defensive check: ensure daily_sales exists before proceeding
    # ---------------------------------------------------------------------
    if sales_all is None:
        logger.warning("⚠️ No daily_sales data found for required months. Skipping.")
        return
    # ---------------------------------------------------------------------
    # Type enforcement (defensive)
    # ---------------------------------------------------------------------
    sales_all = (sales_all
        .withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
        .withColumn("store_nbr", F.col("store_nbr").cast(StringType()))
        .withColumn("family", F.col("family").cast(StringType()))
        .withColumn("sales", F.col("sales").cast(FloatType()))
        .withColumn("onpromotion", F.col("onpromotion").cast(FloatType()))
    )

    if txn_all is not None:
        txn_all = (txn_all
            .withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
            .withColumn("store_nbr", F.col("store_nbr").cast(StringType()))
            .withColumn("transactions", F.col("transactions").cast(IntegerType()))
        )

    if oil_all is not None:
        oil_all = (oil_all
            .withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
            .withColumn("dcoilwtico", F.col("dcoilwtico").cast(FloatType()))
        )

    # ---------------------------------------------------------------------
    # Aggregate daily → monthly
    # ---------------------------------------------------------------------
    sales_m = (sales_all
        .groupBy("snapshot_date", "store_nbr", "family")
        .agg(
            F.sum("sales").alias("sales"),
            F.sum("onpromotion").alias("onpromotion_sum"),
            F.avg("onpromotion").alias("onpromotion_avg")
        )
    )

    if txn_all is not None:
        txn_m = txn_all.groupBy("snapshot_date", "store_nbr") \
                       .agg(F.sum("transactions").alias("transactions"))
    else:
        txn_m = None

    if oil_all is not None:
        oil_m = oil_all.groupBy("snapshot_date") \
                       .agg(F.avg("dcoilwtico").alias("avg_dcoilwtico"))
    else:
        oil_m = None

    # ---------------------------------------------------------------------
    # Join + LM/L3M feature engineering
    # ---------------------------------------------------------------------
    feats_all = sales_m
    if txn_m is not None:
        feats_all = feats_all.join(txn_m, ["snapshot_date", "store_nbr"], "left")
    if oil_m is not None:
        feats_all = feats_all.join(oil_m, ["snapshot_date"], "left")

    # ===== 🧩 Add holiday reference join here =====
    holiday_path = resolve_path(config["directories"].get("holiday_ref", "datamart/silver_reference/holiday_features.parquet"))
    if os.path.exists(holiday_path):
        holiday_df = spark.read.parquet(holiday_path)
        holiday_df = holiday_df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
        feats_all = feats_all.join(holiday_df, on="snapshot_date", how="left")
        logger.info("🎉 Joined holiday features from silver_reference.")
    else:
        logger.warning("⚠️ Holiday features not found, skipping join.")

    # ===== Define windows =====
    w_sf = Window.partitionBy("store_nbr", "family").orderBy("snapshot_date").rowsBetween(-2, 0)
    w_s  = Window.partitionBy("store_nbr").orderBy("snapshot_date").rowsBetween(-2, 0)
    w_o  = Window.orderBy("snapshot_date").rowsBetween(-2, 0)

    # ===== LM / L3M features =====
    feats_all = (feats_all
        .withColumn("sales_LM", F.col("sales"))
        .withColumn("AVG_sales_L3M", F.avg("sales").over(w_sf))
        .withColumn("SUM_sales_L3M", F.sum("sales").over(w_sf))
        .withColumn("onpromotion_LM", F.col("onpromotion_sum"))
        .withColumn("AVG_onpromotion_L3M", F.avg("onpromotion_sum").over(w_sf))
        .withColumn("SUM_onpromotion_L3M", F.sum("onpromotion_sum").over(w_sf))
        .withColumn("transactions_LM", F.col("transactions"))
        .withColumn("AVG_transactions_L3M", F.avg("transactions").over(w_s))
        .withColumn("SUM_transactions_L3M", F.sum("transactions").over(w_s))
        .withColumn("AVG_dcoilwtico_LM", F.col("avg_dcoilwtico"))
        .withColumn("AVG_dcoilwtico_L3M", F.avg("avg_dcoilwtico").over(w_o))
        .withColumn("SUM_dcoilwtico_L3M", F.sum("avg_dcoilwtico").over(w_o))
    )

    # Only retain features for current month (S)
    feats_S = feats_all.filter(F.col("snapshot_date") == F.lit(S))

    # ---------------------------------------------------------------------
    # Label = S sales, written to S-1
    # ---------------------------------------------------------------------
    labels_S_minus_1 = (sales_m
        .filter(F.col("snapshot_date") == F.lit(S))
        .select(
            F.lit(S1).cast(DateType()).alias("snapshot_date"),
            "store_nbr", "family",
            F.col("sales").alias("label_next_month_sales")
        )
    )

    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    def _write_single(df, out_dir, tag):
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, f"tmp_{tag}")
        final = os.path.join(out_dir, f"{tag}.parquet")
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        df.coalesce(1).write.mode("overwrite").parquet(tmp)
        part = glob.glob(os.path.join(tmp, "part-*.parquet"))
        if part:
            shutil.move(part[0], final)
        shutil.rmtree(tmp, ignore_errors=True)
        return final

    tag_S  = fmt_tag(S)
    tag_S1 = fmt_tag(S1)
    feat_path = _write_single(feats_S, gold_feature_dir, tag_S)
    lab_path  = _write_single(labels_S_minus_1, gold_label_dir, tag_S1)

    logger.info(f"✅ Feature store → {feat_path}")
    logger.info(f"✅ Label store   → {lab_path}")
    logger.info("🎉 Gold processing completed successfully.")

# -------------------------------------------------------------------------
# Entrypoint (Airflow / CLI)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Process Gold Tables")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--config_path", default="config/gold_config.yaml", help="Config path")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("GoldLayerProcessing").getOrCreate()
    process_gold_tables(args.snapshot_date, spark, args.config_path)
