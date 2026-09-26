import mlflow
# проверка соединения с mlflow и отправка тестового файла в хранилище S3
# доступы в S3 из .env

mlflow.set_tracking_uri("http://127.0.0.1:3000")
mlflow.set_experiment("test_1") #введи любое название теста

with mlflow.start_run():
    mlflow.log_artifact("train.py")   # или любой другой существующий файл
    
