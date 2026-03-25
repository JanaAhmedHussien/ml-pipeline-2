import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with mlflow.start_run() as run:
    # Load data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    # Log Accuracy
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    # Save Run ID for the next job
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"Model trained with accuracy: {accuracy}")