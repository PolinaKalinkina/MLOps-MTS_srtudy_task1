# ML Scoring Service (MLOps ДЗ)

Сервис упаковывает CatBoost-модель ранжирования в Docker-контейнер, который
принимает `test.csv`, прогоняет препроцессинг и inference и выгружает
`sample_submission.csv`, а также JSON с топ-5 feature importances и график
распределения предсказанных скоров.


## Структура проекта

```
mlops-service/
├── Dockerfile
├── requirements.txt
├── README.md
├── model/
│   └── model.cbm            # обученный артефакт CatBoostRanker
├── src/
│   ├── config.py            # названия колонок и пути — единственное место для правок
│   ├── load_data.py         # этап 1: загрузка test.csv
│   ├── preprocess.py        # этап 2: препроцессинг
│   ├── score.py              # этап 3: inference (CPU only)
│   ├── export.py            # этап 4: выгрузка результатов
│   └── run_pipeline.py      # точка входа контейнера, вызывает этапы 1-4 по очереди
├── train/
│   └── train_baseline.py    # обучение бейзлайна (запускается ЛОКАЛЬНО, не в контейнере)
├── sample_data/
│   └── test.csv              # демонстрационный test.csv для проверки сервиса
├── input/                   # <- сюда монтируется ваш test.csv
└── output/                  # <- сюда попадают результаты скоринга
```

## Что делает сервис

При запуске контейнер выполняет 4 шага **отдельными скриптами**:

1. `load_data.py` — читает `./input/test.csv`, проверяет наличие обязательных
   идентификационных колонок.
2. `preprocess.py` — приводит числовые признаки к float с заполнением
   пропусков медианой, категориальные — к строке с заполнением `"missing"`.
3. `score.py` — загружает `model/model.cbm` и считает предсказания
   (`CatBoostRanker.predict`, только CPU).
4. `export.py` — сохраняет в `./output`:
   - `sample_submission.csv` — id-колонки + предсказанный `score`;
   - `feature_importances_top5.json` — топ-5 фичей по важности;
   - `score_distribution.png` — гистограмма распределения скоров.

## Быстрый старт

### 1. Сборка образа

```bash
docker build -t ml-scoring-service .
```

### 2. Подготовка входных данных

Положите ваш `test.csv` в директорию `input` в корне проекта. Для проверки
работоспособности сервиса "из коробки" можно использовать готовый
демонстрационный файл:

```bash
cp sample_data/test.csv input/test.csv
```

### 3. Запуск контейнера

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  ml-scoring-service
```

### 4. Результат

После завершения работы в директории `output` появятся:

```
output/
├── sample_submission.csv
├── feature_importances_top5.json
└── score_distribution.png
```

Проверить содержимое:

```bash
head output/sample_submission.csv
cat output/feature_importances_top5.json
```

## Формат данных

Демонстрационный датасет имитирует задачу ранжирования (например, для
рекомендательной системы): для каждого `query_id` есть несколько `item_id`
с признаками, модель должна отранжировать их по релевантности.

`test.csv` содержит колонки:

| колонка          | тип         | описание                              |
|------------------|-------------|----------------------------------------|
| `query_id`       | id          | идентификатор запроса/пользователя     |
| `item_id`        | id          | идентификатор объекта                  |
| `feature_1..8`   | float       | числовые признаки                      |
| `feature_cat_1/2`| category    | категориальные признаки                |

`sample_submission.csv` содержит: `query_id`, `item_id`, `score`.

## Как подставить свою модель в пайплайн

Шаблон специально сделан так, чтобы переход на другие данные требовал
минимум правок:

1. **Обучите модель на реальных данных.** Отредактируйте
   `train/train_baseline.py` (или напишите свой скрипт обучения), заменив
   `generate_synthetic_dataset()` на загрузку вашего `train.csv` с реальными
   фичами из соревнования ML 2025.
2. **Обновите `src/config.py`** — впишите реальные названия
   `ID_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`, `TARGET_COLUMN`.
   Остальной код (`load_data.py`, `preprocess.py`, `score.py`, `export.py`)
   менять не нужно — он читает конфигурацию динамически.
3. **Пересохраните артефакт** — запустите обучение локально:
   ```bash
   pip install -r requirements.txt
   python train/train_baseline.py
   ```
   Это перезапишет `model/model.cbm` и `sample_data/test.csv`.
4. Если ваша лучшая модель — не CatBoostRanker, а, например,
   `CatBoostClassifier`/`CatBoostRegressor` или другая библиотека (lightgbm,
   sklearn), поменяйте импорт и вызов `load_model`/`predict` в `src/score.py`
   — остальной пайплайн не затрагивается.
5. Пересоберите Docker-образ (`docker build ...`) — модель уже будет внутри
   образа, обучение внутри контейнера не выполняется (только inference).

## Зависимости

См. `requirements.txt`. Версии зафиксированы для воспроизводимости сборки.

## Известные ограничения

- Inference выполняется только на CPU (GPU не используется, в соответствии
  с требованиями задания).
- Обучение модели происходит вне контейнера (сервис делает только inference,
  как указано в задании).
