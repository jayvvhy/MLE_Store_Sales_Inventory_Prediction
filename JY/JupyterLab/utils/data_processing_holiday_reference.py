import os
import sys
import yaml
import logging
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType, BooleanType, IntegerType

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
# Helper: type map from YAML
# -------------------------------------------------------------------------
def get_spark_type(dtype_str: str):
    mapping = {
        "string": StringType(),
        "date": DateType(),
        "bool": BooleanType(),
        "int": IntegerType(),
    }
    return mapping.get(dtype_str, StringType())

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------
def process_holiday_reference(spark, config_path: str = "config/holiday_events_config.yaml"):
    """
    Process holiday_events.csv into Bronze + Silver reference tables.
    Requires an existing SparkSession.
    """
    # ---------------------------------------------------------------------
    # Load config
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

    raw_data_dir = resolve_path(config["directories"]["raw_data_dir"])
    bronze_dir = resolve_path(config["directories"]["bronze_dir"])
    silver_dir = resolve_path(config["directories"]["silver_dir"])
    schema_cfg = config.get("schema", {})

    os.makedirs(bronze_dir, exist_ok=True)
    os.makedirs(silver_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step 1: Read raw CSV
    # ---------------------------------------------------------------------
    csv_path = os.path.join(raw_data_dir, "holidays_events.csv")
    logger.info(f"📂 Reading {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # ---------------------------------------------------------------------
    # Step 2: Enforce schema from YAML
    # ---------------------------------------------------------------------
    for col_name, dtype_str in schema_cfg.items():
        spark_type = get_spark_type(dtype_str)
        if col_name in df.columns:
            if dtype_str == "bool":
                # Normalize mixed-type booleans (string, bool, int)
                df = (
                    df.withColumn(
                        col_name,
                        F.when(
                            F.lower(F.col(col_name).cast("string")).isin("true", "1", "t", "yes", "y"),
                            F.lit(True)
                        ).otherwise(F.lit(False))
                    )
                )
            df = df.withColumn(col_name, F.col(col_name).cast(spark_type))

    df = df.withColumn("date", F.to_date("date"))
    df = df.withColumn("snapshot_date", F.trunc("date", "month").cast(DateType()))
    df = df.withColumn("transferred", F.col("transferred").cast("int"))

    # ---------------------------------------------------------------------
    # Step 3: Write Bronze reference parquet
    # ---------------------------------------------------------------------
    bronze_out = os.path.join(bronze_dir, "holiday_events_full.parquet")
    df.write.mode("overwrite").parquet(bronze_out)
    logger.info(f"✅ Saved Bronze reference: {bronze_out}")

    # ---------------------------------------------------------------------
    # Step 4: Monthly aggregation
    # ---------------------------------------------------------------------
    types = [r["type"] for r in df.select("type").distinct().collect()]
    locales = [r["locale"] for r in df.select("locale").distinct().collect()]

    agg_exprs = [
        F.count("*").alias("num_holidays_total"),
        F.sum("transferred").alias("num_transferred"),
    ]
    for t in types:
        agg_exprs.append(F.sum(F.when(F.col("type") == t, 1).otherwise(0)).alias(f"num_holidays_type_{t}"))
    for l in locales:
        agg_exprs.append(F.sum(F.when(F.col("locale") == l, 1).otherwise(0)).alias(f"num_holidays_locale_{l}"))

    df_monthly = df.groupBy("snapshot_date").agg(*agg_exprs).orderBy("snapshot_date")

    # ---------------------------------------------------------------------
    # Step 5: Create NM (next-month) shifted features
    # ---------------------------------------------------------------------
    df_nm = df_monthly.select(
        F.add_months(F.col("snapshot_date"), -1).alias("snapshot_date"),
        *[F.col(c).alias(f"{c}_NM") for c in df_monthly.columns if c != "snapshot_date"]
    )

    df_combined = df_monthly.join(df_nm, on="snapshot_date", how="left")

    # ---------------------------------------------------------------------
    # Step 6: Write Silver reference
    # ---------------------------------------------------------------------
    silver_out = os.path.join(silver_dir, "holiday_features.parquet")
    df_combined.write.mode("overwrite").parquet(silver_out)
    logger.info(f"✅ Saved Silver reference: {silver_out}")

    logger.info("🎉 Holiday reference processing completed successfully.")

# -------------------------------------------------------------------------
# Entrypoint (optional CLI)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    from pyspark.sql import SparkSession

    parser = ArgumentParser(description="Process Holiday Reference Tables")
    parser.add_argument("--config_path", default="config/holiday_events_config.yaml")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("HolidayReferenceProcessing").getOrCreate()
    process_holiday_reference(spark, args.config_path)
