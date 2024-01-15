import pickle
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model(df: pd.DataFrame):
    x = df.drop('Survived',axis=1)
    y = df.Survived
    cate_features_index = np.where(x.dtypes != float)[0]
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.85,random_state=1234)
    with mlflow.start_run():
        model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True, random_seed=42)
        model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
        mlflow.log_artifact(local_path="/home/xflow/project/scripts/train_model.py",
                            artifact_path="train_model code")
        mlflow.end_run()
    print('Test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(BASE_DIR, "datasets/train_prep.csv"))
    train_model(df)