from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["alerts@yourdomain.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="monthly_model_pipeline",
    description="Monthly end-to-end ML pipeline: Bronze → Silver → Gold → Train → Promote → Inference → Monitoring",
    default_args=default_args,
    schedule_interval="@monthly",        # runs on the 1st of each month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    tags=["MLE", "model_pipeline"],
) as dag:

    bronze_task = BashOperator(
        task_id="bronze_processing",
        bash_command="python /opt/airflow/utils/data_processing_bronze_table.py --snapshot_date {{ ds }}",
    )

    silver_task = BashOperator(
        task_id="silver_processing",
        bash_command="python /opt/airflow/utils/data_processing_silver_table.py --snapshot_date {{ ds }}",
    )

    gold_task = BashOperator(
        task_id="gold_processing",
        bash_command="python /opt/airflow/utils/data_processing_gold_table.py --snapshot_date {{ ds }}",
    )

    model_train_task = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/utils/model_train.py --snapshot_date {{ ds }}",
    )

    promote_best_task = BashOperator(
        task_id="promote_best_model",
        bash_command="python /opt/airflow/utils/promote_best_model.py --snapshot_date {{ ds }}",
    )

    inference_task = BashOperator(
        task_id="run_inference",
        bash_command="python /opt/airflow/utils/model_inference.py --snapshot_date {{ ds }}",
    )

    monitoring_task = BashOperator(
        task_id="monitor_model",
        bash_command="python /opt/airflow/utils/model_monitoring.py --snapshot_date {{ ds }}",
    )

    update_registry_task = BashOperator(
        task_id="update_model_registry",
        bash_command="python /opt/airflow/utils/update_model_registry.py"
    )

    bronze_task >> silver_task >> gold_task >> promote_best_task >> model_train_task >> update_registry_task >> inference_task >> monitoring_task
