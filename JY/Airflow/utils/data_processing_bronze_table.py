import os
import sys
import glob
import shutil
import yaml
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------
def process_bronze_month(snapshot_date: str, spark: SparkSession, config_path: str = "config/bronze_config.yaml"):
    """
    Process raw CSVs into Bronze Parquet tables for a specific month.

    Args:
        snapshot_date (str): Month start date (YYYY-MM-DD) — typically first day of month.
        spark (SparkSession): Active Spark session.
        config_path (str): Path to YAML configuration file.
    """
    if snapshot_date is None:
        raise ValueError("snapshot_date must be provided (format: YYYY-MM-DD).")

    # ---------------------------------------------------------------------
    # Resolve config path
    # ---------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    resolved_path = os.path.join(base_dir, config_path)
    resolved_path = os.path.abspath(resolved_path)

    logger.info(f"🔍 Using config file: {resolved_path}")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    raw_data_dir = config["raw_data_dir"]
    bronze_dir = config["bronze_dir"]
    datasets = config["datasets"]

    # ---------------------------------------------------------------------
    # Define time window for this snapshot month
    # ---------------------------------------------------------------------
    start_date = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
    end_date = (start_date + relativedelta(months=1)) - timedelta(days=1)
    logger.info(f"🗓️ Processing Bronze layer for month: {start_date} → {end_date}")

    # Use underscore-separated snapshot tag: YYYY_MM_DD
    snapshot_tag = start_date.strftime("%Y_%m_%d")

    # ---------------------------------------------------------------------
    # Process each dataset
    # ---------------------------------------------------------------------
    for filename, table_name in datasets.items():
        csv_path = os.path.join(raw_data_dir, filename)
        logger.info(f"📂 Reading file: {csv_path}")

        if not os.path.exists(csv_path):
            logger.warning(f"⚠️ File not found: {csv_path}. Skipping...")
            continue

        # Read CSV
        df = spark.read.csv(csv_path, header=True, inferSchema=True)

        # Expect a "date" column in the dataset
        if "date" not in df.columns:
            raise ValueError(f"'date' column missing in {filename}")

        # Filter records within the month window
        df = df.withColumn("date", to_date(col("date")))
        df_month = df.filter((col("date") >= start_date.isoformat()) & (col("date") <= end_date.isoformat()))

        count_filtered = df_month.count()
        logger.info(f"📅 {table_name}: {count_filtered} records for {snapshot_tag}")

        if count_filtered == 0:
            logger.warning(f"No records found for {table_name} in {snapshot_tag}. Skipping...")
            continue

        # -----------------------------------------------------------------
        # Define output paths and write to Parquet
        # -----------------------------------------------------------------
        table_output_dir = os.path.join(bronze_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

        temp_output_dir = os.path.join(table_output_dir, f"tmp_{snapshot_tag}")
        final_output_path = os.path.join(table_output_dir, f"{snapshot_tag}.parquet")

        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        (
            df_month
            .coalesce(1)
            .write
            .mode("overwrite")
            .parquet(temp_output_dir)
        )

        # Move the single generated part file into the correct snapshot filename
        parquet_files = glob.glob(os.path.join(temp_output_dir, "part-*.parquet"))
        if not parquet_files:
            logger.error(f"No parquet file generated for {table_name} ({snapshot_tag}).")
            continue

        shutil.move(parquet_files[0], final_output_path)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        logger.info(f"✅ Saved Bronze file: {final_output_path} ({count_filtered} rows)")

    logger.info("🎉 Monthly Bronze processing completed successfully.")
# ---------------------------------------------------------------------
# ✅ Airflow entrypoint
# ---------------------------------------------------------------------
from utils.helper_spark import get_spark_session

def main(snapshot_date: str, spark=None):
    """
    Entry point for Bronze ETL. Creates Spark session if not provided.
    """
    if spark is None:
        spark = get_spark_session("bronze_table")

    process_bronze_month(snapshot_date, spark)

    try:
        spark.stop()
    except Exception:
        pass


# -------------------------------------------------------------------------
# Entrypoint (Airflow / CLI)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    from pyspark.sql import SparkSession

    parser = ArgumentParser(description="Process Bronze Tables")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--config_path", default="config/bronze_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # ✅ Create Spark only when the script is executed directly
    spark = SparkSession.builder.appName("BronzeMonthlyProcessing").getOrCreate()
    process_bronze_month(snapshot_date=args.snapshot_date, spark=spark, config_path=args.config_path)
