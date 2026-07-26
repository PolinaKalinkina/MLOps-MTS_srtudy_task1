"""
Обучение базовой модели.

Этот скрипт НЕ входит в inference-контейнер — он запускается один раз
локально, чтобы получить артефакт model/model.cbm, который потом
упаковывается в сервис.

По умолчанию скрипт генерирует синтетический датасет в духе задачи
ранжирования (query_id/item_id + признаки + релевантность), чтобы весь
пайплайн можно было проверить без реальных данных соревнования.

Чтобы обучить модель на настоящих данных ML 2025:
  1. Замените generate_synthetic_dataset() на pd.read_csv("train.csv")
     с вашими реальными фичами.
  2. Приведите названия колонок в src/config.py в соответствие с реальными.
  3. Запустите этот скрипт заново — он пересоздаст model/model.cbm
     и sample_data/test.csv.
"""

import os

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

np.random.seed(42)

N_QUERIES = 200
ITEMS_PER_QUERY = 10

FEATURE_COLS = [f"feature_{i}" for i in range(1, 9)]
CAT_COLS = ["feature_cat_1", "feature_cat_2"]


def generate_synthetic_dataset() -> pd.DataFrame:
    rows = []
    for q in range(N_QUERIES):
        query_id = f"q_{q}"
        for i in range(ITEMS_PER_QUERY):
            item_id = f"item_{q}_{i}"
            feats = np.random.randn(8)
            cat1 = np.random.choice(["A", "B", "C"])
            cat2 = np.random.choice(["X", "Y"])
            raw_relevance = feats[0] * 2 + feats[1] - feats[2] * 0.5 + np.random.randn() * 0.3
            rows.append([query_id, item_id, *feats, cat1, cat2, raw_relevance])

    columns = ["query_id", "item_id"] + FEATURE_COLS + CAT_COLS + ["raw_relevance"]
    df = pd.DataFrame(rows, columns=columns)

    # Превращаем непрерывный сигнал в градуированную релевантность 0-4 внутри каждой query.
    rank_within_query = df.groupby("query_id")["raw_relevance"].rank(method="first")
    df["relevance"] = ((rank_within_query - 1) / (ITEMS_PER_QUERY - 1) * 4).round().astype(int)
    df = df.drop(columns=["raw_relevance"])
    return df


def train_and_save() -> None:
    df = generate_synthetic_dataset().sort_values("query_id").reset_index(drop=True)

    train_pool = Pool(
        data=df[FEATURE_COLS + CAT_COLS],
        label=df["relevance"],
        group_id=df["query_id"],
        cat_features=CAT_COLS,
    )

    model = CatBoostRanker(
        iterations=200,
        learning_rate=0.1,
        loss_function="YetiRank",
        verbose=False,
        random_seed=42,
    )
    model.fit(train_pool)

    os.makedirs("model", exist_ok=True)
    model.save_model("model/model.cbm")
    print("Модель сохранена в model/model.cbm")

    os.makedirs("sample_data", exist_ok=True)
    test_df = (
        df.drop(columns=["relevance"])
        .sample(frac=0.3, random_state=1)
        .reset_index(drop=True)
    )
    test_df.to_csv("sample_data/test.csv", index=False)
    print(f"Демонстрационный test.csv ({len(test_df)} строк) сохранён в sample_data/test.csv")


if __name__ == "__main__":
    train_and_save()
