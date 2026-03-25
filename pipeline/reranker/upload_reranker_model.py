import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # ajuste se necessário

with mlflow.start_run(run_id="7d67ebad781c48fd87a630ff2745e2e1"):
    mlflow.log_artifacts("models/reranker", artifact_path="model-final")