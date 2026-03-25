import mlflow
import sys

THRESHOLD = 0.85
mlflow.set_tracking_uri("file:./mlruns")
# Read run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy is None:
    print("No accuracy logged. Failing pipeline.")
    sys.exit(1)

if accuracy < THRESHOLD:
    print(f"Accuracy {accuracy} is below threshold {THRESHOLD}. Failing.")
    sys.exit(1)

print("Model passed validation!")