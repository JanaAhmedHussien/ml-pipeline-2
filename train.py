import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI with authentication
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

# For DagsHub, you need to include credentials in the URI
if username and password:
    # Parse the URI to insert credentials
    if "https://" in mlflow_tracking_uri:
        base_uri = mlflow_tracking_uri.replace("https://", "")
        authenticated_uri = f"https://{username}:{password}@{base_uri}"
        mlflow.set_tracking_uri(authenticated_uri)
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
else:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set experiment (optional but recommended)
mlflow.set_experiment("ml-pipeline-experiment")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    df = pd.read_csv("data/train.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    mlflow.sklearn.log_model(model, "model")

    with open("model_info.txt", "w") as f:
        f.write(run_id)