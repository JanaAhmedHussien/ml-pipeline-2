import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

def verify_performance():
    if not os.path.exists("model_info.txt"):
        print("Error: model_info.txt not found!")
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    
    # Fetch data from MLflow
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    
    print(f"Validating Run {run_id} | Accuracy: {accuracy:.4f}")
    
    # TASK: Fail pipeline if accuracy < 0.85
    if accuracy < 0.85:
        print("RESULT: REJECTED (Accuracy below 0.85)")
        sys.exit(1)
    else:
        print("RESULT: PASSED (Proceeding to Deployment)")

if __name__ == "__main__":
    verify_performance()