"""
store_sales_reference_setup.py
------------------------------
One-time setup DAG for reference tables.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import (
    data_processing_holiday_events_reference,
    data_processing_start_dates_reference,
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="store_sales_reference_setup",
    description="One-time setup for reference data tables (holiday events, start dates).",
    default_args=default_args,
    start_date=datetime(2013, 1, 1),
    schedule_interval="@once",  # one-off setup
    catchup=True,
    tags=["store_sales", "reference", "setup"],
) as dag:

    create_holiday_ref = PythonOperator(
        task_id="create_holiday_events_reference",
        python_callable=data_processing_holiday_events_reference.main,
    )

    create_start_date_ref = PythonOperator(
        task_id="create_start_dates_reference",
        python_callable=data_processing_start_dates_reference.main,
    )

    create_holiday_ref >> create_start_date_ref
