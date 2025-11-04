import os
import sys
import yaml
import logging
import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Helper: resolve path (works for both Airflow & Jupyter)
# -------------------------------------------------------------------------
def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    base_dir = "/opt/airflow" if os.path.exists("/opt/airflow") else os.getcwd()
    return os.path.join(base_dir, path_str.strip("/"))

# -------------------------------------------------------------------------
# Main process
# -------------------------------------------------------------------------
def process_start_dates(spark, config_path: str = "config/start_dates_config.yaml"):
    """
    Creates reference tables that record the first active month for:
      - each store_nbr
      - each family
      - each store_nbr × family combination

    Reads directly from raw data/train.csv (full dataset),
    and saves to datamart/silver_reference/.
    """

    # ---------------------------------------------------------------------
    # Load config
    # ---------------------------------------------------------------------
    resolved_path = resolve_path(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = resolve_path(config["directories"]["raw_data_dir"])
    silver_ref_dir = resolve_path(config["directories"]["silver_reference_dir"])
    os.makedirs(silver_ref_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw train.csv not found in: {raw_dir}")

    logger.info(f"🚀 Creating start-date reference tables from {csv_path}")

    # ---------------------------------------------------------------------
    # Read raw data (full historical)
    # ---------------------------------------------------------------------
    df_sales = spark.read.csv(csv_path, header=True, inferSchema=True)
    logger.info(f"✅ Loaded train.csv with {df_sales.count():,} rows")

    # ---------------------------------------------------------------------
    # Prepare date and filter active sales
    # ---------------------------------------------------------------------
    df_sales = (
        df_sales
        .withColumn("date", F.to_date("date"))
        .withColumn("snapshot_month", F.trunc("date", "month").cast(DateType()))
        .withColumn("store_nbr", F.col("store_nbr").cast("string"))
        .withColumn("family", F.col("family").cast("string"))
        .withColumn("sales", F.col("sales").cast("float"))
    )

    # Only consider active records
    df_active = df_sales.filter(F.col("sales") > 0)

    # ---------------------------------------------------------------------
    # Compute first active month per level
    # ---------------------------------------------------------------------
    store_start = (
        df_active.groupBy("store_nbr")
        .agg(F.min("snapshot_month").alias("first_active_month"))
        .orderBy("store_nbr")
    )

    family_start = (
        df_active.groupBy("family")
        .agg(F.min("snapshot_month").alias("first_active_month"))
        .orderBy("family")
    )

    store_family_start = (
        df_active.groupBy("store_nbr", "family")
        .agg(F.min("snapshot_month").alias("first_active_month"))
        .orderBy("store_nbr", "family")
    )

    # ---------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------
    def _write(df, name):
        out_path = os.path.join(silver_ref_dir, name)
        df.write.mode("overwrite").parquet(out_path)
        logger.info(f"💾 Saved {name} → {out_path}")

    _write(store_start, "store_start_dates.parquet")
    _write(family_start, "family_start_dates.parquet")
    _write(store_family_start, "store_family_start_dates.parquet")

    logger.info("🎉 Start-date reference tables created successfully.")

# -------------------------------------------------------------------------
# ✅ Airflow entrypoint (called by DAG)
# -------------------------------------------------------------------------
from utils.helper_spark import get_spark_session

def main(spark=None):
    if spark is None:
        spark = get_spark_session("holiday_events_reference")

    process_start_dates(spark)

    try:
        spark.stop()
    except Exception:
        pass
# -------------------------------------------------------------------------
# Entrypoint (Airflow / CLI)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Store/Family Start Dates Reference")
    parser.add_argument("--config_path", default="config/start_dates_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # ⚠️ Spark should be created externally (not started/stopped here)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("StartDatesReferenceProcessing").getOrCreate()

    process_start_dates(spark=spark, config_path=args.config_path)
