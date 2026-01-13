"""
Мульти-ячеечный dataset builder для обучения на нескольких локациях.

Создаёт объединённый датасет из качественных ячеек с:
- Погодными фичами
- Метками is_flyable
- Confidence-weighted labels для учёта неопределённости
"""

import json
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from paraglideml.data.flight_parsing import load_flights_to_dataframe
from paraglideml.data.weather_cache import WeatherCache


def compute_label_confidence(row: pd.Series) -> float:
    """
    Вычисляет уверенность в метке is_flyable.

    is_flyable = 1 (лётный день):
    - flight_count >= 5: confidence = 1.0 (точно лётный)
    - flight_count 3-4: confidence = 0.9

    is_flyable = 0 (нелётный день):
    - weekend + 0 полётов: confidence = 0.8 (вероятно нелётный)
    - будний + 0 полётов: confidence = 0.5 (неопределённо, люди на работе)
    - flight_count 1-2: confidence = 0.6 (пограничный случай)

    Returns:
        float: confidence weight (0.5-1.0)
    """
    flight_count = row["flight_count"]
    is_flyable = row["is_flyable"]
    is_weekend = row.get("is_weekend", 0)

    if is_flyable == 1:
        # Лётный день
        if flight_count >= 5:
            return 1.0
        else:  # flight_count 3-4
            return 0.9
    else:
        # Нелётный день
        if flight_count == 0:
            if is_weekend:
                return 0.8  # Выходной без полётов - скорее всего плохая погода
            else:
                return 0.5  # Будний без полётов - люди на работе
        else:  # flight_count 1-2
            return 0.6  # Пограничный случай


def build_cell_dataset(
    cell_id: str, df_flights: pd.DataFrame, cache: WeatherCache, min_xc_points: int = 10
) -> pd.DataFrame:
    """
    Создаёт датасет для одной ячейки.

    Args:
        cell_id: строка вида '45_11'
        df_flights: DataFrame со всеми полётами
        cache: экземпляр WeatherCache
        min_xc_points: минимальные баллы XContest для качественного XC полёта

    Returns:
        DataFrame с колонками: date, flight_count, is_flyable, label_confidence,
                               cell_id, cell_lat, cell_lon, [weather_features],
                               is_weekend, day_of_year
    """
    cell_lat, cell_lon = map(int, cell_id.split("_"))

    # Фильтруем полёты в границах ячейки (±0.5° от центра)
    cell_center_lat = cell_lat + 0.5
    cell_center_lon = cell_lon + 0.5
    TOLERANCE = 0.5

    df_geo = df_flights.dropna(subset=["takeoff_lat", "takeoff_lon"])
    df_cell = df_geo[
        (df_geo["takeoff_lat"].between(cell_center_lat - TOLERANCE, cell_center_lat + TOLERANCE))
        & (df_geo["takeoff_lon"].between(cell_center_lon - TOLERANCE, cell_center_lon + TOLERANCE))
    ].copy()

    if len(df_cell) == 0:
        return pd.DataFrame()

    # Фильтруем по качеству XC (баллы XContest)
    if min_xc_points > 0:
        df_cell = df_cell[df_cell["points"] >= min_xc_points].copy()
        if len(df_cell) == 0:
            return pd.DataFrame()

    # Агрегация по дням
    df_cell["date_only"] = pd.to_datetime(df_cell["date"]).dt.tz_localize(None).dt.normalize()
    daily_flights = df_cell.groupby("date_only").agg(flight_count=("id", "count")).reset_index()
    daily_flights.rename(columns={"date_only": "date"}, inplace=True)

    # Создаём target timeline (15 мая - 15 сентября)
    target_dates = []
    available_years = [2021, 2022, 2023, 2024, 2025]
    for y in available_years:
        rng = pd.date_range(f"{y}-05-15", f"{y}-09-15")
        target_dates.extend(rng)

    df_target = pd.DataFrame({"date": target_dates})
    df_target["date"] = pd.to_datetime(df_target["date"]).dt.normalize()
    df_target = df_target.merge(daily_flights, on="date", how="left").fillna(0)
    # После фильтрации по качеству: 1+ качественный полёт = летный день
    df_target["is_flyable"] = (df_target["flight_count"] >= 1).astype(int)

    # Добавляем day_of_week для вычисления is_weekend
    df_target["is_weekend"] = (df_target["date"].dt.dayofweek >= 5).astype(int)

    # Вычисляем confidence для каждой строки
    df_target["label_confidence"] = df_target.apply(compute_label_confidence, axis=1)

    # Мержим с погодными данными
    weather_records = []
    for date in df_target["date"]:
        sample = cache.load_sample(cell_lat, cell_lon, date, 12)
        if sample:
            features = sample["features"]
            features["date"] = date
            weather_records.append(features)

    if not weather_records:
        return pd.DataFrame()

    df_weather = pd.DataFrame(weather_records)
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.normalize()

    # Финальный merge
    dataset = df_target.merge(df_weather, on="date", how="inner")

    # Добавляем метаданные ячейки
    dataset["cell_id"] = cell_id
    dataset["cell_lat"] = cell_lat
    dataset["cell_lon"] = cell_lon
    dataset["day_of_year"] = dataset["date"].dt.dayofyear

    return dataset


def build_multicell_dataset(
    cells: Optional[List[str]] = None,
    flights_dir: str = "data/flights",
    cache_root: str = "data/gfs/cache",
    selected_cells_path: str = "data/processed/selected_cells.json",
    output_path: str = "data/processed/multicell_dataset.csv",
    min_xc_points: int = 10,
) -> pd.DataFrame:
    """
    Создаёт объединённый датасет из нескольких ячеек.

    Args:
        cells: список cell_id вида ['45_11', '46_11', ...].
               Если None, загружается из selected_cells.json
        flights_dir: путь к папке с JSON полётами
        cache_root: корень weather cache
        selected_cells_path: путь к JSON со списком ячеек
        output_path: путь для сохранения CSV
        min_xc_points: минимальные баллы XContest для качественного XC полёта

    Returns:
        pd.DataFrame с объединённым датасетом
    """
    # Загружаем список ячеек
    if cells is None:
        with open(selected_cells_path, "r") as f:
            cells = json.load(f)

    print("Загружаю полёты...")
    df_flights = load_flights_to_dataframe(data_dir=flights_dir)

    print("Инициализирую weather cache...")
    cache = WeatherCache(cache_root=cache_root)

    print(f"\nСоздаю датасеты для {len(cells)} ячеек с фильтром XC>={min_xc_points} очков...\n")

    all_datasets = []
    for cell_id in tqdm(cells, desc="Обработка ячеек"):
        cell_df = build_cell_dataset(cell_id, df_flights, cache, min_xc_points)
        if not cell_df.empty:
            all_datasets.append(cell_df)
        else:
            print(f"  ⚠ Ячейка {cell_id}: нет данных, пропускаем")

    if not all_datasets:
        print("Ошибка: ни одна ячейка не вернула данные!")
        return pd.DataFrame()

    # Объединяем все ячейки
    multicell_df = pd.concat(all_datasets, ignore_index=True)

    # Сохраняем
    multicell_df.to_csv(output_path, index=False)

    # Статистика
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")
    print(f"Всего строк: {len(multicell_df)}")
    print(f"Уникальных дат: {multicell_df['date'].nunique()}")
    print(f"Ячеек в датасете: {multicell_df['cell_id'].nunique()}")
    print("\nБаланс классов:")
    print(multicell_df["is_flyable"].value_counts(normalize=True))
    print(multicell_df["is_flyable"].value_counts())

    print("\nКоличество строк по ячейкам:")
    print(multicell_df["cell_id"].value_counts().head(10))

    print("\nРаспределение label_confidence:")
    print(multicell_df["label_confidence"].describe())

    print(f"\n✓ Датасет сохранён в {output_path}")

    # Сохраняем статистику в JSON
    stats = {
        "total_rows": int(len(multicell_df)),
        "unique_dates": int(multicell_df["date"].nunique()),
        "num_cells": int(multicell_df["cell_id"].nunique()),
        "flyable_ratio": float(multicell_df["is_flyable"].mean()),
        "cells": cells,
        "confidence_stats": {
            "mean": float(multicell_df["label_confidence"].mean()),
            "min": float(multicell_df["label_confidence"].min()),
            "max": float(multicell_df["label_confidence"].max()),
        },
    }

    stats_path = output_path.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Статистика сохранена в {stats_path}")

    return multicell_df


if __name__ == "__main__":
    # Создаём multicell датасет
    # По умолчанию используем отфильтрованные ячейки
    import sys

    selected_path = "data/processed/selected_cells.json"
    output_path = "data/processed/multicell_dataset.csv"

    # Можно передать путь как аргумент
    if len(sys.argv) > 1:
        selected_path = sys.argv[1]

    df = build_multicell_dataset(selected_cells_path=selected_path, output_path=output_path)

    if not df.empty:
        print("\nПример строки датасета:")
        print(df.head(1).T)
