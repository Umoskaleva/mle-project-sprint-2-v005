"""
Отдельный скрипт для логирования уже обученной модели в MLflow + S3.
Предполагает, что train_model_model.py уже отработал и создал:
  - model.pkl (или model.cb)
  - results/model_signature.json
  - results/cv_mean_scores.json
  - results/test_metrics.json
  - results/test_predictions.csv
  - data/X_train.csv, X_test.csv, y_train.csv, y_test.csv
  - params.yaml
  - python_env.yaml
  - requirements.txt
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml


# ---------- Конфигурация ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:3000")
EXPERIMENT_NAME = "churn_logreg_uliana"
REGISTERED_MODEL_NAME = "churn_logreg_uliana"

MODEL_PATH = "model.cb"                    
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

MODEL_SIGNATURE_PATH = RESULTS_DIR / "model_signature.json"
CV_SCORES_PATH = RESULTS_DIR / "cv_mean_scores.json"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

PARAMS_PATH = "params.yaml"
PYTHON_ENV_PATH = "python_env.yaml"
REQUIREMENTS_PATH = "requirements.txt"
TRAIN_SCRIPT_PATH = "src/train_model_model.py"


def log_run():
    # ---------- 1. Настройка MLflow ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # ---------- 2. Загрузка модели ----------
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Модель загружена: {MODEL_PATH} ({type(model).__name__})")

    # ---------- 3. Загрузка данных и сигнатуры ----------
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    with open(MODEL_SIGNATURE_PATH, "r", encoding="utf-8") as f:
        signature_dict = json.load(f)

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # ---------- 4. Загрузка метрик ----------
    with open(CV_SCORES_PATH, "r", encoding="utf-8") as f:
        cv_scores = json.load(f)

    with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    # ---------- 5. Логирование ----------
    with mlflow.start_run() as run:
        print(f"\nRun ID: {run.info.run_id}")

        # 5.1 Параметры
        model_hyperparams = params["model"]["hyperparams"]
        mlflow.log_params({
            "model_type": type(model).__name__,
            "max_iter": model_hyperparams.get("max_iter"),
            "C": model_hyperparams.get("C"),
            "random_state": params["split"]["random_state"],
            "test_size": params["split"]["test_size"],
            "cv_n_splits": params["cv"]["n_splits"],
            "target_col": params["split"]["target_col"],
        })

        # 5.2 Метрики CV
        mlflow.log_metrics({f"cv_{k}": float(v) for k, v in cv_scores.items()})

        # 5.3 Метрики теста (только скаляры)
        scalar_metrics = {
            k: float(v) for k, v in test_metrics.items() if k != "confusion_matrix"
        }
        mlflow.log_metrics({f"test_{k}": v for k, v in scalar_metrics.items()})

        # 5.4 Логирование модели
        # Используем mlflow.sklearn, т.к. у вас LogisticRegression.
        # Если модель CatBoost — замените на mlflow.catboost.log_model.
        # Если хотите универсально — используйте mlflow.pyfunc.log_model.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=mlflow.models.ModelSignature.from_dict(signature_dict),
            input_example=X_train.iloc[:5],
            pip_requirements=PYTHON_ENV_PATH,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Модель залогирована: {model_info.model_uri}")

        # 5.5 Дополнительные артефакты
        mlflow.log_artifact(TRAIN_SCRIPT_PATH, artifact_path="code")
        mlflow.log_artifact(PARAMS_PATH, artifact_path="config")
        mlflow.log_artifact(REQUIREMENTS_PATH, artifact_path="environment")
        mlflow.log_artifact(str(CV_SCORES_PATH), artifact_path="metrics")
        mlflow.log_artifact(str(TEST_METRICS_PATH), artifact_path="metrics")
        mlflow.log_artifact(str(TEST_PREDICTIONS_PATH), artifact_path="predictions")
        mlflow.log_artifact(str(MODEL_SIGNATURE_PATH), artifact_path="model")

        # 5.6 Теги (удобно для фильтрации в UI)
        mlflow.set_tags({
            "author": os.getenv("USER", "unknown"),
            "source": "log_to_mlflow.py",
            "framework": "sklearn",
        })

        print(f"\n✅ Готово. Откройте UI: {MLFLOW_TRACKING_URI}")
        return run.info.run_id


if __name__ == "__main__":
    log_run()