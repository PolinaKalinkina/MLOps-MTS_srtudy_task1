"""
Этап 1: загрузка входного файла test.csv из примонтированной директории ./input
"""

import os
import pandas as pd

from config import INPUT_DIR, TEST_FILENAME, ID_COLUMNS


def load_test_data() -> pd.DataFrame:
    path = os.path.join(INPUT_DIR, TEST_FILENAME)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден входной файл: {path}. "
            f"Поместите test.csv в примонтированную директорию ./input"
        )

    df = pd.read_csv(path)

    missing_ids = [c for c in ID_COLUMNS if c not in df.columns]
    if missing_ids:
        raise ValueError(
            f"В test.csv отсутствуют обязательные идентификационные колонки: {missing_ids}"
        )

    print(f"[load_data] Загружено {len(df)} строк, {len(df.columns)} колонок из {path}")
    return df


if __name__ == "__main__":
    load_test_data()
