# Проект: churn_model_uliana (mle-project-sprint-2-v005)

## Описание
Проект по работе с моделью предсказания оттока клиентов (**churn classification**).
Модель обучается, логируется и регистрируется в **MLflow**, который поднимается локально.
Артефакты модели хранятся в **S3-совместимом хранилище** (например, Yandex Object Storage),
откуда их можно скачать в локальную папку `saved_model/` для дальнейшей работы.

## Структура проекта
```text
mle-project-sprint-2-v005/
├── mlflow_local/                          # локальный MLflow: SQLite backend и артефакты
│   ├── mlflow.db
│   └── artifacts/
│
├── notebooks/
│   └── test_mlflow_connection.ipynb       # проверка подключения к MLflow
│
├── saved_model/                           # артефакты обученной модели
│   ├── conda.yaml
│   ├── input_example.json                 # пример входных данных для signature и метрик
│   ├── MLmodel
│   ├── model.skops                        # обученная модель v1
│   ├── python_env.yaml
│   ├── requirements.txt                   # зависимости модели
│   └── serving_input_example.json         # пример запроса для inference-сервера
│
├── scripts/
│   ├── run_mlflow_server_local.sh         # запуск MLflow: SQLite + локальные артефакты
│   └── run_mlflow_server_postgresql.sh    # запуск MLflow: PostgreSQL + S3
│
├── src/
│   ├── load_model.py                      # скачивание модели из S3 в saved_model/
│   ├── logging_registration_model.py      # логирование и регистрация baseline-модели
│   └── logging_registration_model_retrain.py  # переобучение, метрики и новая версия в Registry
│
├── .env                                   # S3 endpoint и credentials (не коммитится)
├── .gitignore
├── params.yaml                            # параметры модели и обучения
├── python_env.yaml
├── README.md
└── requirements.txt
```

## Установка
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Создайте .env с переменными:
MLFLOW_TRACKING_URI=<URL MLflow-сервера>
MLFLOW_S3_ENDPOINT_URL=<URL S3>
AWS_ACCESS_KEY_ID=<ключ>
AWS_SECRET_ACCESS_KEY=<секрет>


## Запуск MLflow локально
```bash
mlflow server \
  --backend-store-uri sqlite:///./mlflow_local/mlflow.db \
  --default-artifact-root file:./mlflow_local/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

# Основные сценарии
1. Скачивание артефактов модели из S3 и загрузка модели
Скрипт src/load_model.py:

читает креды из .env;

подключается к S3 через boto3 по указанному endpoint_url;

рекурсивно скачивает все файлы из заданного prefix в saved_model/;

загружает модель через mlflow.sklearn.load_model.

2. Логирование и регистрация модели в MLflow
Скрипт src/logging_registration_model.py:

подключается к локальному MLflow (127.0.0.1:5000);

загружает модель из saved_model/;

строит signature через infer_signature на основе input_example.json;

создаёт (при необходимости) эксперимент churn_uliana;

в рамках run model_0_registry логирует:

метрики по input_example (n_samples_logged, n_positive_pred, positive_rate);

параметры модели (model.get_params());

саму модель с сигнатурой, примером входа, requirements.txt и metadata;

регистрирует модель в Model Registry под именем churn_model_uliana.

3. Переобучение новой версии модели
Скрипт src/logging_registration_model_retrain.py:

обучает новую версию модели (v2);

считает метрики (ROC-AUC, Precision, Recall, F1, Accuracy);

логирует и регистрирует новую версию в MLflow;

позволяет сравнить v1 и v2 в одном эксперименте.




## 1. Постановка задачи

Требовалось:

1. Обучить новую версию модели `churn_model_uliana` (v2).
2. Зарегистрировать её в реестре моделей MLflow.
3. Посчитать метрики: ROC-AUC, Precision, Recall, F1, Accuracy.
4. Сравнить v1 и v2 между собой.

## 2. Исходные данные и их ограничения

### 2.1. Что было доступно

В артефактах папки `saved_model/` присутствовали:

| Файл | Назначение |
|------|-----------|
| `model.skops` | обученная модель v1 (baseline) |
| `input_example.json` | пример входных данных — 5 строк, 20 признаков |
| `MLmodel`, `conda.yaml`, `requirements.txt`, `python_env.yaml` | метаданные и окружение |
| `serving_input_example.json` | пример для сервинга |

### 2.2. Чего не было
* **`X_train`, `y_train`** — обучающая выборка.
* **`X_test`, `y_test`** — тестовая выборка.

## 3. Принятый подход

### 3.1. Восстановление структуры данных

Из `input_example.json` извлечена **структура признаков**:

* **20 колонок**, из них:
  * **3 непрерывные**: `monthly_charges`, `total_charges`, `senior_citizen`
    (значения стандартизированы — `StandardScaler`);
  * **17 бинарных**: One-Hot-закодированные категориальные признаки
    (`type_*`, `payment_method_*`, `internet_service_*`, `online_*`,
    `device_*`, `tech_support_*`, `streaming_*`, `gender_*`, `partner_*`,
    `dependents_*`, `multiple_lines_*`, `paperless_billing_*`).

### 3.2. Генерация синтетического датасета

Сгенерировано **2000 строк** с сохранением структуры:

* непрерывные признаки — из `N(0, 1)` (соответствует стандартизированным данным);
* бинарные признаки — из распределения Бернулли с частотой единиц, взятой
  из `input_example.json` (с ограничением `[0.05, 0.95]`);
* целевая переменная `y_synth` — по эвристике на основе линейной комбинации
  признаков с последующим логистическим преобразованием. Это необходимо,
  чтобы модель могла чему-то научиться и метрики не были случайными
  (иначе ROC-AUC ≈ 0.5).

Датасет сохранён:
saved_model/synthetic/
├── synthetic_full.csv # 2000 строк
├── synthetic_train.csv # 1600 строк (80%)
└── synthetic_test.csv # 400 строк (20%)