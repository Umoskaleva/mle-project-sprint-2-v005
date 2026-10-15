"""
Задание:
  1. Загрузить уже обученную модель churn_model_uliana из локальной папки saved_model/.
  2. Залогировать её в MLflow-эксперимент churn_uliana под именем запуска model_0_registry.
  3. Зарегистрировать модель в Model Registry MLflow под именем churn_model_uliana.

Что делает код:
  - Подключается к локальному MLflow-серверу (tracking + registry) по адресу 127.0.0.1:5000.
  - Загружает модель из saved_model/ через mlflow.sklearn.load_model.
  - Читает два JSON-примера входных данных:
      * input_example.json         — для signature, input_example и расчёта метрик;
      * serving_input_example.json — для проверки/отправки запроса к inference-серверу MLflow.
  - Строит сигнатуру модели через infer_signature на основе input_example.
  - Указывает requirements.txt как pip-зависимости модели.
  - Формирует metadata: тип модели, задачу (churn_classification), источник.
  - Создаёт эксперимент churn_uliana, если его ещё нет.
  - В рамках run:
      * считает метрики по input_example: n_samples_logged, n_positive_pred, positive_rate;
      * логирует параметры модели (model.get_params());
      * логирует модель с сигнатурой, input_example, requirements и metadata,
        регистрируя её в Model Registry как churn_model_uliana.
"""


import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature


# === 1. Константы ===
EXPERIMENT_NAME = "churn_uliana"
RUN_NAME = "model_0_registry"
REGISTRY_MODEL_NAME = "churn_model_uliana"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

# === 2. Путь к уже скачанной модели ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "saved_model"

# === 3. Подключение к локальному MLflow ===
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# === 4. Загружаем модель и примеры ===
model = mlflow.sklearn.load_model(str(LOCAL_MODEL_DIR))

#  для signature, input_example и расчёта метрик в run
with open(LOCAL_MODEL_DIR / "input_example.json", "r") as f: 
    input_example = json.load(f)

# для проверки/отправки запроса к inference-серверу MLflow
with open(LOCAL_MODEL_DIR / "serving_input_example.json", "r") as f: # 
    serving_input_example = json.load(f)

# === 5. Signature, requirements, metadata ===
X_example = pd.DataFrame(
    input_example["data"],
    columns=input_example["columns"],
)

signature = infer_signature(
    model_input=X_example,
    model_output=model.predict(X_example),
)

pip_requirements = str(LOCAL_MODEL_DIR / "requirements.txt")

metadata = {
    "model_type": type(model).__name__,
    "task": "churn_classification",
    "source": "local saved_model/",
}

# === 6. Эксперимент: создать, если нет ===
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# === 7. Логируем run ===
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # --- Метрики по input_example ---
    X = pd.DataFrame(
    input_example["data"],
    columns=input_example["columns"],
)
    y_pred = model.predict(X)
    mlflow.log_metric("n_samples_logged", len(X))
    mlflow.log_metric("n_positive_pred", int((y_pred == 1).sum()))
    mlflow.log_metric("positive_rate", float((y_pred == 1).mean()))

    # --- Параметры модели ---
    mlflow.log_params(model.get_params())

    # --- Логирование модели + регистрация ---
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        metadata=metadata,
        registered_model_name=REGISTRY_MODEL_NAME,
    )

    