import os
import gdown
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/xflow/project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("data_download")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_dataset(id):
    gdown.download(id=id, output=os.path.join(BASE_DIR, "datasets/train.csv"))

if __name__ == "__main__":
    with mlflow.start_run():
        download_dataset('1XOQnXIyYjNlgu2V1QGwu9U3_gnSY2R7C')
    mlflow.log_artifact(local_path="/home/xflow/project/scripts/data_download.py",
                        artifact_path="data_download code")
    mlflow.end_run()
