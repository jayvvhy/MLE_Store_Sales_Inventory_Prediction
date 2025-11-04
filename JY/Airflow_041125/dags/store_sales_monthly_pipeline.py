"""
store_sales_monthly_pipeline.py
-------------------------------
Main monthly backfill DAG for ETL + Model pipeline.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import (
    data_processing_bronze_table,
    data_processing_silver_table,
    data_processing_gold_table,
    model_train,
    model_inference,
    update_model_registry,
    model_monitoring,
    promote_best_model,
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="store_sales_monthly_pipeline",
    description="Monthly ETL + Model Training + Monitoring pipeline (backfilled)",
    default_args=default_args,
    start_date=datetime(2013, 1, 1),
    end_date=datetime(2017, 7, 1),
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["store_sales", "MLE", "pipeline"],
) as dag:

    bronze = PythonOperator(
        task_id="bronze",
        python_callable=data_processing_bronze_table.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    silver = PythonOperator(
        task_id="silver",
        python_callable=data_processing_silver_table.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    gold = PythonOperator(
        task_id="gold",
        python_callable=data_processing_gold_table.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    train = PythonOperator(
        task_id="model_train",
        python_callable=model_train.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    inference = PythonOperator(
        task_id="model_inference",
        python_callable=model_inference.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    registry = PythonOperator(
        task_id="update_model_registry",
        python_callable=update_model_registry.main,
    )

    monitor = PythonOperator(
        task_id="model_monitoring",
        python_callable=model_monitoring.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    promote = PythonOperator(
        task_id="promote_best_model",
        python_callable=promote_best_model.main,
        op_kwargs={"snapshot_date": "{{ ds }}"},
    )

    bronze >> silver >> gold >> promote >> train >> inference >> registry >> monitor 
