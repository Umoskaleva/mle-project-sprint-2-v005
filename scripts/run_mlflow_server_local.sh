mlflow server \
  --backend-store-uri sqlite:///./mlflow_local/mlflow.db \
    --default-artifact-root file:./mlflow_local/artifacts
    --host 127.0.0.1 \
    --port 5000