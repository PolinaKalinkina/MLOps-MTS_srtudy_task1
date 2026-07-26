"""
Централизованная конфигурация сервиса.

Если вы адаптируете этот шаблон под свои реальные данные из соревнования
ML 2025, вам нужно поменять только значения в этом файле (названия колонок),
остальной код менять не придётся.
"""

import os

# Идентификационные колонки, которые обязательно должны быть в test.csv
# и которые пробрасываются в submission без изменений.
ID_COLUMNS = ["query_id", "item_id"]

# Колонка группировки для ранжирования (обычно совпадает с query_id / user_id).
GROUP_COLUMN = "query_id"

# Числовые и категориальные признаки, на которых обучена модель.
NUMERIC_COLUMNS = [f"feature_{i}" for i in range(1, 9)]
CATEGORICAL_COLUMNS = ["feature_cat_1", "feature_cat_2"]

# Целевая переменная (используется только при обучении, в test.csv её нет).
TARGET_COLUMN = "relevance"

# Пути внутри контейнера (примонтированные директории).
INPUT_DIR = os.getenv("INPUT_DIR", "/app/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.cbm")

# Имена файлов.
TEST_FILENAME = "test.csv"
SUBMISSION_FILENAME = "sample_submission.csv"
FEATURE_IMPORTANCE_FILENAME = "feature_importances_top5.json"
SCORE_DIST_FILENAME = "score_distribution.png"
