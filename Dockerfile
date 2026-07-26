FROM python:3.11-slim

WORKDIR /app

# Ставим зависимости отдельным слоем, чтобы кешировался при неизменном requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код сервиса и уже обученный артефакт модели (только inference, обучение не требуется)
COPY src/ ./src/
COPY model/ ./model/

# Директории для примонтированных volume'ов
RUN mkdir -p /app/input /app/output

WORKDIR /app/src
CMD ["python", "run_pipeline.py"]
