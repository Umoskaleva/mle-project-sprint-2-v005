"""
Задание:
  1. Обучить новую версию модели churn_model_uliana (v2).
  2. Зарегистрировать churn_model_uliana в реестре моделей MLflow.
  3. Посчитать метрики: ROC-AUC, Precision, Recall, F1, Accuracy.
  4. Сравнить v1 и v2.

Ограничение:
  Исходные X_train / X_test / y_train / y_test недоступны.
  В артефактах есть только:
    - saved_model/model.skops          (обученная модель v1)
    - saved_model/input_example.json   (пример входных данных, 5 строк)
  Поэтому генерируется синтетический датасет, повторяющий структуру примера.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score,
)
from mlflow.models import infer_signature

# ============================================================
# 1. Константы
# ============================================================
EXPERIMENT_NAME     = "churn_uliana"
RUN_NAME            = "model_1_registry"
REGISTRY_MODEL_NAME = "churn_model_uliana"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "saved_model"

MODEL_V1_PATH   = MODEL_DIR / "model.skops"
EXAMPLE_PATH    = MODEL_DIR / "input_example.json"
MODEL_V2_PATH   = MODEL_DIR / "model_v2.skops"
METRICS_V1_PATH = MODEL_DIR / "metrics_v1.json"
METRICS_V2_PATH = MODEL_DIR / "metrics_v2.json"
COMPARISON_PATH = MODEL_DIR / "comparison.csv"

SYNTH_DIR        = MODEL_DIR / "synthetic"
SYNTH_FULL_PATH  = SYNTH_DIR / "synthetic_full.csv"
SYNTH_TRAIN_PATH = SYNTH_DIR / "synthetic_train.csv"
SYNTH_TEST_PATH  = SYNTH_DIR / "synthetic_test.csv"

RANDOM_STATE = 42
N_SAMPLES    = 2000
TEST_SIZE    = 0.2

# ============================================================
# 1a. Подключение к локальному MLflow-серверу
# ============================================================
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# ============================================================
# 2. Загружаем пример входных данных
# ============================================================
with open(EXAMPLE_PATH, "r") as f:
    example = json.load(f)

columns   = example["columns"]
X_example = pd.DataFrame(example["data"], columns=columns)

# ============================================================
# 3. Загружаем модель v1
# ============================================================
unknown_types = sio.get_untrusted_types(file=str(MODEL_V1_PATH))
model_v1 = sio.load(str(MODEL_V1_PATH), trusted=unknown_types)

# ============================================================
# 4. Генерируем синтетический датасет
# ============================================================
np.random.seed(RANDOM_STATE)

continuous_cols = []
binary_cols     = []
for col in columns:
    values = X_example[col].unique()
    if set(values).issubset({0.0, 1.0}):
        binary_cols.append(col)
    else:
        continuous_cols.append(col)

X_synth = pd.DataFrame()
for col in continuous_cols:
    X_synth[col] = np.random.normal(0, 1, N_SAMPLES)

for col in binary_cols:
    p = X_example[col].mean()
    p = float(np.clip(p, 0.05, 0.95))
    X_synth[col] = np.random.binomial(1, p, N_SAMPLES).astype(float)

X_synth = X_synth[columns]

def _col(df, name):
    return df[name] if name in df.columns else 0.0

logits = (
    0.5 * _col(X_synth, "monthly_charges")
    + 0.3 * _col(X_synth, "internet_service_Fiber optic")
    + 0.2 * _col(X_synth, "payment_method_Electronic check")
    - 0.4 * _col(X_synth, "type_Two year")
    - 0.3 * _col(X_synth, "tech_support_Yes")
)
probs   = 1 / (1 + np.exp(-logits))
y_synth = (np.random.rand(N_SAMPLES) < probs).astype(int)

# ============================================================
# 5. Train/test split
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X_synth, y_synth,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_synth,
)

# ============================================================
# 6. Сохраняем синтетику на диск
# ============================================================
os.makedirs(SYNTH_DIR, exist_ok=True)

X_synth.assign(target=y_synth).to_csv(SYNTH_FULL_PATH,  index=False)
X_tr.assign(target=y_tr).to_csv(SYNTH_TRAIN_PATH, index=False)
X_te.assign(target=y_te).to_csv(SYNTH_TEST_PATH,  index=False)

# ============================================================
# 7. Обучаем новую модель v2
# ============================================================
model_v2 = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model_v2.fit(X_tr, y_tr)

sio.dump(model_v2, str(MODEL_V2_PATH))

# ============================================================
# 8. Функция подсчёта метрик
# ============================================================
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
    }

# ============================================================
# 9. Метрики v1 и v2 на одном и том же тесте
# ============================================================
y_pred_v1  = model_v1.predict(X_te)
y_proba_v1 = model_v1.predict_proba(X_te)[:, 1]
metrics_v1 = compute_metrics(y_te, y_pred_v1, y_proba_v1)

y_pred_v2  = model_v2.predict(X_te)
y_proba_v2 = model_v2.predict_proba(X_te)[:, 1]
metrics_v2 = compute_metrics(y_te, y_pred_v2, y_proba_v2)

with open(METRICS_V1_PATH, "w") as f:
    json.dump(metrics_v1, f, indent=2)
with open(METRICS_V2_PATH, "w") as f:
    json.dump(metrics_v2, f, indent=2)

# ============================================================
# 10. Сравнение моделей
# ============================================================
comparison = pd.DataFrame({
    "v1_old": metrics_v1,
    "v2_new": metrics_v2,
})
comparison["delta"] = (comparison["v2_new"] - comparison["v1_old"]).round(4)
comparison.to_csv(COMPARISON_PATH)

# ============================================================
# 11. Регистрация модели в MLflow Model Registry
# ============================================================
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "model_type":       type(model_v2).__name__,
        "n_estimators":     model_v2.n_estimators,
        "max_depth":        model_v2.max_depth,
        "min_samples_leaf": model_v2.min_samples_leaf,
        "n_samples_train":  len(X_tr),
        "data_source":      "synthetic",
    })

    mlflow.log_metrics({f"v1_{k}": v for k, v in metrics_v1.items()})
    mlflow.log_metrics({f"v2_{k}": v for k, v in metrics_v2.items()})

    mlflow.log_artifact(str(COMPARISON_PATH), artifact_path="comparison")
    mlflow.log_artifact(str(METRICS_V1_PATH), artifact_path="comparison")
    mlflow.log_artifact(str(METRICS_V2_PATH), artifact_path="comparison")

    signature = infer_signature(X_tr, model_v2.predict(X_tr))

    mlflow.sklearn.log_model(
        sk_model=model_v2,
        name="model",
        signature=signature,
        input_example=X_example,
        registered_model_name=REGISTRY_MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )