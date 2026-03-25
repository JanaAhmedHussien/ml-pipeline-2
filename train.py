import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Configure MLflow with Secrets
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("Assignment_Experiment")

def train_model():
    # Load data (In a real DVC setup, you'd load 'data.csv' here)
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    
    with mlflow.start_run() as run:
        # Use Random Forest as requested
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # Log to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # TASK: Export Run ID to model_info.txt
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)
        
        print(f"Successfully logged Run: {run.info.run_id} with Accuracy: {acc}")

if __name__ == "__main__":
    train_model()