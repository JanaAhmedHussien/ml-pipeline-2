import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


with mlflow.start_run() as run:
    run_id = run.info.run_id

    df = pd.read_csv("data/train.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    with open("model_info.txt", "w") as f:
        f.write(run_id)