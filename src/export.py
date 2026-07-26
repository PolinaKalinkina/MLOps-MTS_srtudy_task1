"""
Этап 4: выгрузка результатов в примонтированную директорию ./output.

Формирует:
  - sample_submission.csv               (обязательно, зачёт на 4)
  - feature_importances_top5.json       (доп. задание на 5)
  - score_distribution.png              (доп. задание на 5)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRanker

from config import (
    OUTPUT_DIR,
    SUBMISSION_FILENAME,
    FEATURE_IMPORTANCE_FILENAME,
    SCORE_DIST_FILENAME,
    ID_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)


def export_submission(scored_df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = scored_df[ID_COLUMNS + ["score"]]
    path = os.path.join(OUTPUT_DIR, SUBMISSION_FILENAME)
    submission.to_csv(path, index=False)
    print(f"[export] Сохранён {path}")


def export_feature_importances(model: CatBoostRanker) -> None:
    feature_cols = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    importances = model.get_feature_importance(type="PredictionValuesChange")
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:5]
    top5 = {name: float(value) for name, value in pairs}

    path = os.path.join(OUTPUT_DIR, FEATURE_IMPORTANCE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(top5, f, ensure_ascii=False, indent=2)
    print(f"[export] Сохранён {path}")


def export_score_distribution(scored_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(scored_df["score"], bins=40, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.title("Распределение предсказанных скоров")
    plt.xlabel("score")
    plt.ylabel("количество объектов")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, SCORE_DIST_FILENAME)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[export] Сохранён {path}")
