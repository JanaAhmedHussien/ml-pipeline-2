import mlflow
import os
import sys

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Fetch accuracy from MLflow
client = mlflow.tracking.MlflowClient()
data = client.get_run(run_id).data
accuracy = data.metrics.get("accuracy")

print(f"Checking Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print(" Accuracy below threshold! Failing pipeline.")
    sys.exit(1)
else:
    print("Accuracy threshold met. Proceeding to deploy.")
    sys.exit(0)