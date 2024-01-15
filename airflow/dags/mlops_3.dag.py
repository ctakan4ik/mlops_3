from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2023, 12, 7, 15, 15),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False,
}

with DAG(
    "Titanic-Training",
    description="Titanic binary classification",
    schedule_interval="*/1 * * * *",
    default_args=args,
    tags=["titanic", "classification"],
) as dag:
    data_download = BashOperator(
        task_id="data_download",
        bash_command="python3 /home/xflow/project/scripts/data_download.py",
        dag=dag,
    )
    data_prep = BashOperator(
        task_id="data_prep",
        bash_command="python3 /home/xflow/project/scripts/data_prep.py",
        dag=dag,
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python3 /home/xflow/project/scripts/train_model.py",
        dag=dag,
    )
    data_download >> data_prep >> train_model
