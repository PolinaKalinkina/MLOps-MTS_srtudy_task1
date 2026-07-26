"""
Точка входа сервиса. Последовательно выполняет все этапы:
загрузка -> препроцессинг -> скоринг -> выгрузка результатов.
"""

from load_data import load_test_data
from preprocess import preprocess
from score import load_model, score
from export import export_submission, export_feature_importances, export_score_distribution


def main() -> None:
    print("=== Запуск сервиса скоринга ===")

    raw_df = load_test_data()
    processed_df = preprocess(raw_df)

    model = load_model()
    scored_df = score(processed_df, model)

    export_submission(scored_df)
    export_feature_importances(model)
    export_score_distribution(scored_df)

    print("=== Готово. Результаты в ./output ===")


if __name__ == "__main__":
    main()
