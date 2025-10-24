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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window


def process_labels_gold_table(snapshot_date_str, silver_directory, gold_label_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    partition_name = "silver_holiday_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df_h = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_h.count())

    df_gold = df.join(df_h.select("date", "locale"), on="date", how="left")
    df_gold = df_gold.withColumn(
    "is_holiday",
    F.when(F.col("locale").isNotNull(), F.lit(1)).otherwise(F.lit(0))).drop("locale")

    window_spec = (
    Window.partitionBy("store_nbr", "family")
          .orderBy("date")
          .rowsBetween(-2, 0)
    )

    df_gold = (df_gold
    .withColumn("avg_sales_3d", F.avg("sales").over(window_spec))
    .withColumn("sum_sales_3d", F.sum("sales").over(window_spec))
    )

    # # get customer at mob
    # df = df.filter(col("mob") == mob)

    # # get label
    # df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    # df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # # select columns to save
    # df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    output_path = "/opt/airflow/datamart/gold/"
    print(f"Saving gold dataset to: {output_path}")

    df_gold.write.mode("append").parquet(output_path)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df_gold