from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import os
import glob

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a week',
    schedule_interval='0 0 * * 1',  # At 00:00 on day-of-month 1
    start_date=datetime(2013, 1, 1),
    end_date=datetime(2013, 3, 31),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = BashOperator(
    task_id='dep_check_source_label_data',
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 check_source_label_data.py --snapshotdate "{{ ds }}"'
    ),
)

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    def verify_gold_label_store():
        gold_path = "/opt/airflow/datamart/gold/"
        parquet_files = glob.glob(os.path.join(gold_path, "*.parquet"))
    
        if not parquet_files:
            raise FileNotFoundError(f"❌ No Gold label store files found in {gold_path}")
    
        print(f"✅ Found {len(parquet_files)} Gold file(s):")
        for f in parquet_files:
            print(f"  - {os.path.basename(f)}")
        print("✅ Gold Label Store verification completed successfully.")

    label_store_completed = PythonOperator(
    task_id="label_store_completed",
    python_callable=verify_gold_label_store,
    )

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
 
    # --- feature store ---

    gold_feature_store = BashOperator(
        task_id='run_gold_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially

    label_store_completed >> gold_feature_store >> feature_store_completed


    # --- model inference ---
    model_train = BashOperator(
        task_id='run_model_train',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_train.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_train

    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = DummyOperator(task_id="model_1_monitor")

    model_2_monitor = DummyOperator(task_id="model_2_monitor")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_train >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = DummyOperator(task_id="model_1_automl")

    model_2_automl = DummyOperator(task_id="model_2_automl")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed