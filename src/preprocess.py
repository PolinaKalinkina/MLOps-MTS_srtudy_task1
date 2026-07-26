"""
Этап 2: препроцессинг данных.

Базовая логика (сознательно простая — по заданию цель отработать Docker,
а не сложный feature engineering):
  - числовые признаки приводятся к float, пропуски заполняются медианой;
  - категориальные признаки приводятся к строке, пропуски заполняются
    константой "missing" (CatBoost умеет работать с категориями "как есть");
  - если какого-то ожидаемого признака в test.csv нет, он создаётся
    с нейтральным значением, чтобы сервис не падал на неполных данных.
"""

import pandas as pd

from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, ID_COLUMNS


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0.0)

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].astype(str).fillna("missing")

    keep_cols = [c for c in ID_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns]

    print(f"[preprocess] Признаки, подаваемые в модель: {NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}")
    return df[keep_cols]
