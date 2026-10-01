import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

import os
import boto3
from pathlib import Path


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    make_scorer,
)
import joblib  # для сохранения модели


# ---------- Построение scoring-словаря для CV ----------
def build_scoring_dict(metrics_list):
    scoring = {}
    for metric in metrics_list:
        if metric == "roc_auc":
            scoring["roc_auc"] = "roc_auc"
        elif metric == "f1":
            scoring["f1"] = "f1"
        elif metric == "precision":
            scoring["precision"] = "precision"
        elif metric == "recall":
            scoring["recall"] = "recall"
        elif metric in ("logloss", "neg_log_loss"):
            scoring["neg_log_loss"] = "neg_log_loss"
        else:
            raise ValueError(f"Неизвестная метрика для CV: {metric}")
    return scoring


# ---------- Метрики на тесте ----------
def compute_test_metrics(y_true, y_pred, y_proba, metrics_list):
    res = {}
    for m in metrics_list:
        if m == "roc_auc":
            res["roc_auc"] = roc_auc_score(y_true, y_proba)
        elif m == "f1":
            res["f1"] = f1_score(y_true, y_pred, average="macro")
        elif m == "precision":
            res["precision"] = precision_score(y_true, y_pred, average="macro")
        elif m == "recall":
            res["recall"] = recall_score(y_true, y_pred, average="macro")
        elif m == "logloss":
            res["logloss"] = log_loss(y_true, y_proba)
        elif m == "confusion_matrix":
            res["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        else:
            raise ValueError(f"Неизвестная метрика для теста: {m}")
    return res


# ---------- Конвертация pandas dtype → MLflow type ----------
def pandas_dtype_to_mlflow(dtype) -> str:
    s = str(dtype)
    if s.startswith("int"):
        return "long"
    if s.startswith("float"):
        return "double"
    if s.startswith("bool"):
        return "boolean"
    return "string"


# ---------- Ручная сборка MLflow-сигнатуры (fallback) ----------
def build_mlflow_signature(X_train: pd.DataFrame) -> dict:
    inputs = [
        {"name": col, "type": pandas_dtype_to_mlflow(X_train[col].dtype)}
        for col in X_train.columns
    ]
    outputs = [{"name": "prediction", "type": "long"}]
    return {"inputs": inputs, "outputs": outputs, "params": None}


# ---------- main ----------
def main():
    # 1. Чтение params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    target_col = params["split"]["target_col"]
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    model_params = params["model"]["hyperparams"]
    cv_params = params["cv"]

    # 2. Загрузка данных из CSV
    data = pd.read_csv("data/clean_users_churn_for_model_preprocessed.csv")

    if target_col not in data.columns:
        raise ValueError(f"Целевая колонка '{target_col}' не найдена в датасете")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 3. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    # 4. Модель логистической регрессии
    model = LogisticRegression(
        max_iter=model_params["max_iter"],
        C=model_params["C"],
        random_state=random_state,
    )

    # 5. Кросс-валидация
    cv = StratifiedKFold(
        n_splits=cv_params["n_splits"], shuffle=True, random_state=random_state,
    )
    scoring = build_scoring_dict(cv_params["metrics"])
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        n_jobs=cv_params["n_jobs"], return_train_score=False,
    )

    cv_mean_scores = {}
    for metric in cv_params["metrics"]:
        key = f"test_{metric}"
        score = float(np.mean(cv_results[key]))
        if metric == "neg_log_loss":
            cv_mean_scores["logloss"] = -score
        else:
            cv_mean_scores[metric] = score

    print("\n=== CV mean scores ===")
    for metric, value in cv_mean_scores.items():
        print(f"{metric}: {value:.4f}")

    # 6. Обучение финальной модели на train
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 7. Метрики на тесте
    test_metrics = compute_test_metrics(
        y_test, y_pred, y_proba, cv_params["test_metrics"]
    )

    print("\n=== Test metrics ===")
    for metric, value in test_metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:\n{np.array(value)}")
        else:
            print(f"{metric}: {value:.4f}")

    # 8. Сохранение результатов
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "cv_mean_scores.json", "w", encoding="utf-8") as f:
        json.dump(cv_mean_scores, f, indent=2)

    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    results_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    results_df.to_csv(results_dir / "test_predictions.csv", index=False)

    # Сохранение модели (как было — model.cb)
    joblib.dump(model, "model.cb")

    print("\nРезультаты сохранены в папку results/")
    print("Модель сохранена: model.cb")

    # ========== 9. Сигнатура модели (формат MLflow) ==========
    print("\n" + "=" * 50)
    print("СИГНАТУРА МОДЕЛИ (MLflow format)")
    print("=" * 50)

    sample_input = X_train.iloc[:5]
    sample_predict = model.predict(sample_input)
    sample_predict_proba = model.predict_proba(sample_input)

    print(f"\nВход (Input):")
    print(f"  Тип: pandas.DataFrame")
    print(f"  Количество колонок: {X_train.shape[1]}")
    print(f"  Колонки: {list(X_train.columns)}")
    print(f"  Типы данных:")
    for col in X_train.columns:
        print(f"    - {col}: {X_train[col].dtype}")

    print(f"\nВыход (Output):")
    print(f"  predict():")
    print(f"    Тип: numpy.ndarray")
    print(f"    Форма: {sample_predict.shape}")
    print(f"    Пример: {sample_predict}")
    print(f"    Тип значений: {sample_predict.dtype}")

    print(f"\n  predict_proba():")
    print(f"    Тип: numpy.ndarray")
    print(f"    Форма: {sample_predict_proba.shape}")
    print(f"    Пример:\n{sample_predict_proba}")
    print(f"    Тип значений: {sample_predict_proba.dtype}")

    # ----- Пытаемся построить сигнатуру через MLflow (предпочтительно) -----
    mlflow_signature_dict = None
    try:
        from mlflow.models import infer_signature
        signature_obj = infer_signature(X_train, model.predict(X_train))
        mlflow_signature_dict = signature_obj.to_dict()
        print("\nСигнатура построена через mlflow.models.infer_signature()")
    except Exception as e:
        print(f"\nНе удалось использовать infer_signature ({e}).")
        print("Строим сигнатуру вручную из dtypes...")
        mlflow_signature_dict = build_mlflow_signature(X_train)

    # ----- Сохраняем сигнатуру в JSON -----
    with open(results_dir / "model_signature.json", "w", encoding="utf-8") as f:
        json.dump(mlflow_signature_dict, f, indent=2, ensure_ascii=False)

    print(f"\nСигнатура сохранена: results/model_signature.json")
    print("Формат: MLflow (inputs / outputs / params)")
    print("Пример содержимого:")
    print(json.dumps(mlflow_signature_dict, indent=2, ensure_ascii=False)[:500] + "...")
    print("=" * 50)


if __name__ == "__main__":
    main()