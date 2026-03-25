FROM python:3.12-slim

ARG RUN_ID

WORKDIR /app

COPY . .

RUN pip install mlflow scikit-learn

CMD echo "Downloading model for Run ID: ${RUN_ID}"