"""
Задание:
  1. Скачать артефакты обученной модели churn_model_uliana из S3-хранилища.
  2. Сохранить их в локальную папку saved_model/.
  3. Загрузить модель из локальной папки через MLflow.

Что делает код:
  - Определяет корень проекта и загружает переменные окружения из .env.
  - Создаёт S3-клиент через boto3, используя креды и endpoint из переменных окружения
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL, AWS_DEFAULT_REGION).
  - Задаёт bucket и prefix — путь к артефактам конкретной модели в S3
    (27/models/m-745fb6a378424856b9496d3198ebddcc/artifacts/).
  - Рекурсивно скачивает все файлы по этому prefix в локальную папку saved_model/,
    сохраняя структуру подкаталогов и создавая недостающие директории.
  - Загружает скачанную модель из saved_model/ через mlflow.sklearn.load_model.
"""


import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
import mlflow

# 1. Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# 2. S3-клиент
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "ru-central1"),
)

# 3. Пути
bucket = os.getenv("S3_BUCKET_NAME")
prefix = "27/models/m-745fb6a378424856b9496d3198ebddcc/artifacts/"
local_dir = PROJECT_ROOT / "saved_model"

# 4. Скачиваем все файлы из S3 в saved_model/
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel = os.path.relpath(key, prefix)
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))

# 5. Загружаем модель из локальной папки
model = mlflow.sklearn.load_model(str(local_dir))