# utils/helper_spark.py
from pyspark.sql import SparkSession

def get_spark_session(app_name="store_sales_pipeline"):
    """Create or retrieve a SparkSession (shared configuration)."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
