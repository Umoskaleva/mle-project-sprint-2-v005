# Подготовьте данные
# Создайте файл clean_users_charn.csv с вашими данными
# или измените блок загрузки данных под ваш формат

# Запустите обучение
python train_model.py

# MLOps Project: Churn Prediction

## 📌 О проекте
Пайплайн машинного обучения для предсказания оттока клиентов. Включает обучение модели логистической регрессии, кросс-валидацию, оценку метрик и логирование результатов в MLflow (с хранением артефактов в S3).

## 📂 Структура
```text
├── data/               # Данные (raw и preprocessed)
├── notebooks/          # Jupyter ноутбуки
├── results/            # Метрики, предикты, сигнатуры
├── scripts/            # Вспомогательные скрипты (запуск MLflow)
├── src/                # Исходный код
│   ├── train_model_model.py   # Обучение и оценка модели
│   └── log_to_mlflow.py       # Логирование в MLflow
├── model.cb            # Обученная модель
├── params.yaml         # Гиперпараметры и настройки
├── python_env.yaml     # Окружение для MLflow
└── requirements.txt    # Зависимости
```

## ⚙️ Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Создайте `.env` с переменными:
```env
MLFLOW_TRACKING_URI=<URL MLflow-сервера>
MLFLOW_S3_ENDPOINT_URL=<URL S3>
AWS_ACCESS_KEY_ID=<ключ>
AWS_SECRET_ACCESS_KEY=<секрет>
```

## 🚀 Запуск

**1. Обучение модели:**
```bash
python src/train_model_model.py
```
Загружает данные, проводит CV, обучает модель, сохраняет метрики и сигнатуру в `results/`.

**2. Логирование в MLflow:**
```bash
python src/log_to_mlflow.py
```
Отправляет параметры, метрики, модель и артефакты на MLflow-сервер, регистрирует модель.

## 🛠 Стек
Python 3.10+

Scikit-learn (LogisticRegression, метрики)

MLflow (трекинг, реестр моделей)

Pandas / NumPy (обработка данных)

Boto3 (работа с S3)

PyYAML (конфигурация)

Joblib (сериализация модели)

## 📊 MLflow
Tracking Server: Разворачивается локально или удаленно (скрипт scripts/run_mlflow_server_postgresql.sh).
Backend Store: PostgreSQL (для метаданных).
Artifact Store: S3 (для хранения артефактов и моделей).
- UI: адрес задаётся через `MLFLOW_TRACKING_URI`
После запуска скриптов откройте MLflow UI (обычно http://127.0.0.1:3000), чтобы просмотреть эксперименты, сравнить метрики и управлять версиями модели.

## 📝 Конфигурация
Основные настройки проекта вынесены в params.yaml:
split: параметры разделения данных (target_col, test_size, random_state).
model.hyperparams: гиперпараметры логистической регрессии (C, max_iter).
cv: настройки кросс-валидации (n_splits, n_jobs, метрики).

## MLflow и Инфраструктура
