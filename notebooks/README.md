# Paraglideml Notebooks

Пошаговое исследование и разработка модели прогнозирования летных условий.

## Пайплайн разработки

```
Flight Data → Weather Data → Baseline → Neural Net → MultiRegional
```

## Ноутбуки

| Ноутбук | Описание | Основные результаты |
|---------|----------|---------------------|
| **01_flight_data_analysis** | Загрузка и анализ полётных логов XContest | Парсинг JSON, агрегация по дням, фильтрация по локациям |
| **02_weather_data_analysis** | Анализ GFS данных из GRIB2 файлов | Изучение вертикальных профилей, проверка кэша |
| **03_model_baseline** | Baseline модель (RandomForest) | Macro F1 ~0.70, первый бенчмарк |
| **04_neural_network** | Нейросеть (PyTorch FlyNet) | Macro F1 ~0.77, улучшение на +7% |
| **05_multiregional_model** | Мультирегиональная модель | Macro F1 ~0.80, адаптация к регионам |

## Запуск

```bash
# Установить ядро
ipython kernel install --user --name=paraglideml

# Запустить Jupyter
jupyter lab
```

## Структура данных

Каждый ноутбук использует данные из `data/`:

- `data/flights/` — JSON файлы с полётами
- `data/gfs/cache/` — NPZ кэш GFS данных
- `data/processed/` — CSV датасеты для обучения
