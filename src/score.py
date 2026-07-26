"""
Этап 3: скоринг обработанного датасета обученной моделью (только inference, CPU).
"""

import pandas as pd
from catboost import CatBoostRanker

from config import MODEL_PATH, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def load_model() -> CatBoostRanker:
    model = CatBoostRanker()
    model.load_model(MODEL_PATH)
    return model


def score(df: pd.DataFrame, model: CatBoostRanker) -> pd.DataFrame:
    feature_cols = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    preds = model.predict(df[feature_cols])

    result = df.copy()
    result["score"] = preds
    print(f"[score] Посчитаны предсказания для {len(result)} строк")
    return result
