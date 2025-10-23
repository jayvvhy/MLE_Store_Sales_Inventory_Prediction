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

    # # augment data: add month on book
    # df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # # augment data: add days past due
    # df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    # df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    # df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    


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