FROM python:3.10-slim

# Accept the Run ID from the pipeline
ARG RUN_ID
ENV MODEL_RUN_ID=${RUN_ID}

WORKDIR /app

# Simulate downloading the model
RUN echo "Downloading MLflow model artifacts for Run ID: ${MODEL_RUN_ID}" > model_status.txt

CMD ["cat", "model_status.txt"]