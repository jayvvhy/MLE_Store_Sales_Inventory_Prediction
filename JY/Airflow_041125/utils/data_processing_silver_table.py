import os
import sys
import glob
import shutil
import yaml
import logging
import argparse
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

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
# Helper: path resolution (works for Airflow or local)
# -------------------------------------------------------------------------
def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    base_dir = "/opt/airflow" if os.path.exists("/opt/airflow") else os.getcwd()
    return os.path.join(base_dir, path_str.strip("/"))

# -------------------------------------------------------------------------
# Cleaning / transformation helpers
# -------------------------------------------------------------------------
def add_date_columns(df: DataFrame, date_col: str = "date") -> DataFrame:
    """
    Adds standard date-derived columns for all datasets containing a 'date' column:
      - snapshot_date (first day of the month)
      - year (numeric)
      - month (numeric)
    """
    df = df.withColumn(date_col, F.to_date(F.col(date_col)))
    df = df.withColumn("snapshot_date", F.trunc(F.col(date_col), "month"))
    df = df.withColumn("year", F.year(F.col(date_col)))
    df = df.withColumn("month", F.month(F.col(date_col)))
    return df

def impute_oil_prices(df: DataFrame) -> DataFrame:
    """
    Impute missing oil prices by monthly average.
    Uses 'snapshot_date' as the monthly grouping key.
    """
    monthly_mean = (
        df.groupBy("snapshot_date")
          .agg(F.avg("dcoilwtico").alias("mean_price"))
    )

    df = (
        df.join(monthly_mean, on="snapshot_date", how="left")
          .withColumn(
              "dcoilwtico",
              F.when(F.col("dcoilwtico").isNotNull(), F.col("dcoilwtico"))
               .otherwise(F.col("mean_price"))
          )
          .drop("mean_price")
    )
    return df

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------
def process_silver_tables(snapshot_date: str, spark: SparkSession, config_path: str = "config/silver_config.yaml"):
    """
    Converts Bronze tables into cleaned Silver tables.
    Adds snapshot_date, year, month; imputes oil prices;
    and casts columns per YAML config.
    """
    if snapshot_date is None:
        raise ValueError("snapshot_date must be provided (format: YYYY-MM-DD).")

    # ---------------------------------------------------------------------
    # Resolve config path
    # ---------------------------------------------------------------------
    if not os.path.isabs(config_path):
        base_dir = "/opt/airflow" if os.path.exists("/opt/airflow") else os.getcwd()
        resolved_path = os.path.join(base_dir, config_path)
    else:
        resolved_path = config_path
    resolved_path = os.path.abspath(resolved_path)

    logger.info(f"🔍 Using config: {resolved_path}")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    bronze_dir = resolve_path(config["directories"]["bronze_dir"])
    silver_dir = resolve_path(config["directories"]["silver_dir"])
    snapshot_date_str = snapshot_date.replace("-", "_")

    logger.info(f"🚀 Starting Silver processing for snapshot_date={snapshot_date}")

    # ---------------------------------------------------------------------
    # Process each dataset
    # ---------------------------------------------------------------------
    for table_name, dataset_cfg in config["datasets"].items():
        logger.info(f"📂 Processing dataset: {table_name}")

        bronze_path = os.path.join(bronze_dir, table_name, f"{snapshot_date_str}.parquet")
        if not os.path.exists(bronze_path):
            logger.warning(f"⚠️ No Bronze file found for {table_name} ({snapshot_date}). Skipping...")
            continue

        df = spark.read.parquet(bronze_path)

        # Drop nulls if requested
        for key in dataset_cfg.get("drop_nulls", []):
            if key in df.columns:
                df = df.filter(F.col(key).isNotNull())

        # ---------------------------------------------------------------
        # Apply dataset-specific transformations
        # ---------------------------------------------------------------
        if table_name in ["daily_sales", "daily_transactions", "holiday_events", "oil_prices"]:
            if "date" in df.columns:
                df = add_date_columns(df, date_col="date")

            if table_name == "oil_prices":
                df = impute_oil_prices(df)

        # ---------------------------------------------------------------
        # Type casting from YAML config
        # ---------------------------------------------------------------
        type_map = {
            "string": StringType(),
            "int": IntegerType(),
            "float": FloatType(),
            "date": DateType(),
            "bool": BooleanType(),
        }

        for col_name, dtype_str in dataset_cfg.get("types", {}).items():
            if col_name in df.columns:
                df = df.withColumn(col_name, F.col(col_name).cast(type_map[dtype_str]))

        # ---------------------------------------------------------------
        # Write Silver parquet output
        # ---------------------------------------------------------------
        table_output_dir = os.path.join(silver_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

        temp_output_dir = os.path.join(table_output_dir, f"tmp_{snapshot_date_str}")
        final_output_path = os.path.join(table_output_dir, f"{snapshot_date_str}.parquet")

        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        (
            df.coalesce(1)
              .write
              .mode("overwrite")
              .parquet(temp_output_dir)
        )

        parquet_files = glob.glob(os.path.join(temp_output_dir, "part-*.parquet"))
        if parquet_files:
            shutil.move(parquet_files[0], final_output_path)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            logger.info(f"✅ Saved Silver file: {final_output_path}")
        else:
            logger.warning(f"⚠️ No parquet file written for {table_name} ({snapshot_date})")

    logger.info("🎉 Silver layer processing completed successfully.")

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

    process_silver_tables(snapshot_date, spark)

    try:
        spark.stop()
    except Exception:
        pass

# -------------------------------------------------------------------------
# Entrypoint (Airflow / CLI compatible)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    from pyspark.sql import SparkSession

    parser = ArgumentParser(description="Process Silver Tables")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--config_path", default="config/silver_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # ✅ Create SparkSession only when run directly
    spark = SparkSession.builder.appName("SilverLayerProcessing").getOrCreate()
    process_silver_tables(snapshot_date=args.snapshot_date, spark=spark, config_path=args.config_path)

