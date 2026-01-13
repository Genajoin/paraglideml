# Paraglideml

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paraglideml** — ML-система прогнозирования летных условий для парапланеризма на основе метеорологических данных GFS (Global Forecast System) и исторических полётов с XContest.

## Проект

Цель проекта — предсказывать flyability (летные условия) для конкретных локаций: **Flyable / Not Flyable**.

Текущий фокус: Альпийский регион (Словения, Италия, Австрия), модель архитектурируется как универсальная для любых горных и равнинных сайтов.

![Alps Region](docs/alps.png)

---

## Быстрый старт (Example Data)

Для быстрой проверки работоспособности на включенном в репозиторий примере данных старта Kobala (Словения):

```bash
pip install -e .
cp .env.example .env
paraglideml train model
```

## Получение данных (Data Acquisition)

Для полноценной работы системы (вне режима примера) необходимо подготовить следующие данные:

1. **Погода (GFS)**: Скачайте архивы GFS Analysis (0.25 degree) в формате `.grb2`.
   - Источник: [NOAA GFS S3](https://noaa-gfs-bdp-pds.s3.amazonaws.com/)
   - Путь в проекте: `data/gfs/anl/YYYY-MM/` (настраивается в `.env`)
   - Файлы: `gfsanl_3_YYYYMMDD_HH00_000.grb2`

2. **Полеты (XContest)**: Экспортируйте данные о полетах за интересующий период и местность в формате `.json`.
   - Источник: [XContest](https://www.xcontest.org/)
   - Путь в проекте: `data/flights/` (настраивается в `.env`)

Можно использовать инструменты скачивания из проекта [PyParaglide](https://github.com/Genajoin/PyParaglide)

## Полный пайплайн

Проект использует **CLI-first** подход. Все основные операции выполняются через команду `paraglideml`.

```bash
# Установка
pip install -e .

# Проверка конфигурации
paraglideml info

# Шаги пайплайна:
# 1. Подготовка GFS данных
paraglideml data gfs

# 2. Анализ полетов и выбор качественных ячеек
paraglideml data flights

# 3. Сборка обучающего датасета
paraglideml data build

# 4. Обучение модели
paraglideml train model
```

---

## Пайплайн проекта

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ПОДГОТОВКА ДАННЫХ                                            │
├─────────────────────────────────────────────────────────────────┤
│ • Кэширование GFS: paraglideml data gfs                         │
│   → Извлекает 135+ параметров из GRIB2 в NPZ                    │
│   → Использует настройки дат и региона из .env                  │
│                                                                 │
│ • Анализ ячеек: paraglideml data flights                        │
│   → Вычисляет качество ячеек (flights, coverage)                │
│   → Создает data/processed/selected_cells.json                  │
│                                                                 │
│ • Сбор датасета: paraglideml data build                         │
│   → Объединяет weather + flights в multicell_dataset.csv        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ОБУЧЕНИЕ МОДЕЛИ                                              │
├─────────────────────────────────────────────────────────────────┤
│ • MultiRegional Model: paraglideml train model                  │
│   → Архитектура: Regional Attention + Confidence Weighting      │
│   → Оптимизирует Macro F1 Score                                 │
│   → Результат: models/experiments/exp_XXX/                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. АНАЛИЗ РЕЗУЛЬТАТОВ                                           │
├─────────────────────────────────────────────────────────────────┤
│ • Сводка:  paraglideml analyze summary [exp_XXX]                │
│             (по умолчанию — последний эксперимент)              │
│ • Ошибки:  paraglideml analyze errors [exp_XXX]                 │
│             (по умолчанию — последний эксперимент)              │
│ • Сравнение: paraglideml analyze compare --limit 5              │
│ • Ноутбуки: notebooks/05_multiregional_model.ipynb              │
│                                                                 │
│ Артефакты (в папке эксперимента):                               │
│   ├── model.pth, config.json, report.txt                        │
│   ├── training_history.png, confusion_matrix.png                │
│   ├── tp.csv, tn.csv, fp.csv, fn.csv                            │
│   └── per_cell_stats.csv                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Модель: Multi-Regional Attention

Актуальная версия модели (`src/paraglideml/multiregional.py`) решает проблему географической вариативности условий.

**Ключевые особенности:**
- **K-means кластеризация** — группирует ячейки в регионы на основе координат.
- **Regional Embedding** — обучаемые вектора, кодирующие специфику региона.
- **Multi-head Attention** — механизм внимания, адаптирующий общие метео-признаки под конкретный регион.
- **Confidence Weighting** — взвешивание примеров при обучении на основе уверенности в метке (лётный/нелётный день).

Подробнее: `docs/multiregional.md`

---

## Структура проекта

```
paraglideml/
├── .env                          # Конфигурация путей и параметров
├── pyproject.toml
├── README.md
├── MODEL.md                      # Документация по признакам
├── CLAUDE.md / GEMINI.md         # Инструкции для AI агентов
│
├── src/paraglideml/
│   ├── __init__.py
│   ├── cli.py                    # Точка входа CLI (typer)
│   ├── config.py                 # Управление конфигурацией
│   ├── multiregional.py          # Архитектура модели и утилиты
│   ├── train.py                  # Пайплайн обучения
│   │
│   ├── data/                     # Обработка данных
│   │   ├── gfs_processor.py      # Обработка GRIB2 -> NPZ
│   │   ├── cell_analyzer.py      # Анализ ячеек
│   │   ├── dataset_builder.py    # Сборка датасета
│   │   ├── flight_parsing.py     # Парсинг XContest
│   │   └── weather_cache.py      # Чтение NPZ кэша
│   │
│   └── analysis/                 # Инструменты анализа
│       ├── summary.py            # Отчеты и сравнение
│       └── error_analyzer.py     # Детальный анализ ошибок
│
├── notebooks/                    # Jupyter ноутбуки
│   └── 05_multiregional_model.ipynb # Визуальный анализ и эксперименты
│
├── models/                       # Результаты обучения
│   └── experiments/              # Эксперименты (exp_XXX/)
│
├── scripts/                      # Вспомогательные скрипты
│   └── archive/                  # Устаревшие версии
│
└── data/                         # Данные
    ├── gfs/anl/                  # Исходные GRIB2 файлы
    ├── gfs/cache/                # Обработанный NPZ кэш
    ├── flights/                  # Логи полетов (JSON)
    └── processed/                # CSV датасеты и метаданные
```

---

## Конфигурация

Все настройки (пути, параметры обучения, диапазоны дат) управляются через файл `.env` в корне проекта.
Для просмотра текущей конфигурации используйте:

```bash
paraglideml info
```

---

## Разработка

```bash
# Форматирование кода
black src/
isort src/

# Запуск в режиме разработки
pip install -e .
```

---

## License

MIT