# dags/dag.py
import sys, os
sys.path.append("/opt/airflow")  # allow imports from /opt/airflow/utils etc.

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from dateutil.relativedelta import relativedelta

from utils.helper_paths import resolve_relative_path
from utils.helper_spark import get_spark_session
import importlib.util


# --------------------------------------------------------------------------
# Global Spark reference (so we start it once, reuse it across tasks)
# --------------------------------------------------------------------------
_spark_ref = None


def get_or_create_spark():
    """Create or reuse a single SparkSession for this DAG run."""
    global _spark_ref
    if _spark_ref is None:
        _spark_ref = get_spark_session("store_sales_pipeline")
        print("✅ Spark session created once for this DAG run")
    return _spark_ref


def stop_spark():
    """Stop SparkSession at the end of the DAG."""
    global _spark_ref
    if _spark_ref is not None:
        _spark_ref.stop()
        print("🛑 Spark session stopped after DAG run")
        _spark_ref = None


# --------------------------------------------------------------------------
# Dynamically import and run script’s main()
# --------------------------------------------------------------------------
def run_module(script_rel_path, snapshot_date=None, use_spark=False):
    abs_path = resolve_relative_path(script_rel_path)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    spark = get_or_create_spark() if use_spark else None

    if not hasattr(module, "main"):
        raise RuntimeError(f"{script_rel_path} missing main() function")

    if snapshot_date and spark:
        module.main(snapshot_date, spark)
    elif snapshot_date:
        module.main(snapshot_date)
    elif spark:
        module.main(spark)
    else:
        module.main()


# --------------------------------------------------------------------------
# DAG definition
# --------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="store_sales_monthly_pipeline",
    description="Monthly ETL + Model pipeline (shared SparkSession)",
    default_args=default_args,
    start_date=datetime(2013, 1, 1),
    schedule_interval="@monthly",   # run once per month
    catchup=True,                   # backfill older months if needed
    tags=["store_sales", "MLE", "spark"],
) as dag:

    # 1️⃣  One-time reference tables
    holiday_ref = PythonOperator(
        task_id="holiday_reference",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/data_processing_holiday_events_reference.py",
            "use_spark": True,
        },
    )

    startdate_ref = PythonOperator(
        task_id="start_dates_reference",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/data_processing_start_dates_reference.py",
            "use_spark": True,
        },
    )

    # 2️⃣  Monthly ETL + Model tasks (executed for each month)
    bronze = PythonOperator(
        task_id="bronze",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/data_processing_bronze.py",
            "snapshot_date": "{{ ds }}",
            "use_spark": True,
        },
    )

    silver = PythonOperator(
        task_id="silver",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/data_processing_silver.py",
            "snapshot_date": "{{ ds }}",
            "use_spark": True,
        },
    )

    gold = PythonOperator(
        task_id="gold",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/data_processing_gold.py",
            "snapshot_date": "{{ ds }}",
            "use_spark": True,
        },
    )

    promote = PythonOperator(
        task_id="promote_best_model",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/promote_best_model.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    train = PythonOperator(
        task_id="model_train",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/model_train.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    registry = PythonOperator(
        task_id="update_model_registry",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/update_model_registry.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    inference = PythonOperator(
        task_id="model_inference",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/model_inference.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    monitor = PythonOperator(
        task_id="model_monitoring",
        python_callable=run_module,
        op_kwargs={
            "script_rel_path": "utils/model_monitoring.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    close_spark = PythonOperator(
        task_id="stop_spark_session",
        python_callable=stop_spark,
    )

    # 3️⃣  Task dependencies
    [holiday_ref, startdate_ref] >> bronze >> silver >> gold >> promote >> train >> registry >> inference >> monitor >> close_spark
