FROM python:3.12-slim

# TASK: Accept RUN_ID as ARG
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# TASK: Include command to "download" the model
RUN echo "Model successfully downloaded from MLflow Run: ${RUN_ID}" > deployment_status.txt

CMD ["cat", "deployment_status.txt"]