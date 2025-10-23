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


def process_bronze_table(date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/train.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('date') == snapshot_date)
    print(date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_" + date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    csv_holiday_file_path = "data/holidays_events.csv"

    df_h = spark.read.csv(csv_holiday_file_path, header=True, inferSchema=True).filter(col('date') == snapshot_date)
    print(date_str + 'row count:', df.count())

    partition_name = "bronze_holiday_" + date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df_h.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df
