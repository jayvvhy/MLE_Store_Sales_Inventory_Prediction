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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType


def process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "date": DateType(),
        "store_nbr": IntegerType(),
        "family": StringType(),
        "sales": FloatType(),
        "onpromotion": IntegerType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    df = df.filter(
    (df.store_nbr.isin(44, 3, 47)) &
    (df.family.isin("BREAD/BAKERY", "BEVERAGES", "MEATS")))

    # --- (⭐ NEW) Split and save each store-family combination ---
    combos = (
        df.select("store_nbr", "family")
          .distinct()
          .collect()
    )

    for row in combos:
        store_id = row["store_nbr"]
        fam = row["family"].replace("/", "").replace(" ", "").lower()

        sub_df = df.filter((col("store_nbr") == store_id) & (col("family") == row["family"]))
        count = sub_df.count()
        if count == 0:
            continue

        out_name = f"silver_store{store_id}_{fam}_{date_str.replace('-', '_')}.parquet"
        out_path = os.path.join(silver_loan_daily_directory, out_name)

        sub_df.write.mode("overwrite").parquet(out_path)
        print(f"✅ Saved {count} rows to {out_path}")

    partition_name = "bronze_holiday_" + date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df_h = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    df_h = df_h.filter(df_h.locale == "National")

    column_type_map = {
        "date": DateType(),
        "type": StringType(),
        "locale": StringType(),
        "locale_name": StringType(),
        "description": StringType(),
        "transferred": BooleanType()
    }

    for column, new_type in column_type_map.items():
        df_h = df_h.withColumn(column, col(column).cast(new_type))

    # add file saving for holiday

    partition_name = "silver_holiday_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df_h.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    return df