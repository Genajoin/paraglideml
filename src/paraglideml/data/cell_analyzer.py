"""
Анализ качества ячеек для мульти-ячеечного обучения.

Для каждой ячейки вычисляет:
- total_flights: общее количество полётов
- flyable_days: дней с flight_count >= 3
- weather_coverage: % дней с погодными данными (май-сентябрь)
- years_available: список годов с данными
- quality_score: метрика качества для сортировки
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from paraglideml.data.dataset_builder import SEASON_END, SEASON_START
from paraglideml.data.flight_parsing import load_flights_to_dataframe, load_world_flights
from paraglideml.data.weather_cache import WeatherCache


def get_cell_from_coords(lat: float, lon: float) -> Tuple[int, int]:
    """Преобразует координаты в идентификатор ячейки."""
    return int(np.floor(lat)), int(np.floor(lon))


def analyze_cell_flights(
    df_flights: pd.DataFrame, cell_lat: int, cell_lon: int, min_xc_points: int = 10
) -> Dict:
    """
    Анализирует полёты в одной ячейке.

    Args:
        df_flights: DataFrame со всеми полётами
        cell_lat, cell_lon: координаты ячейки
        min_xc_points: минимальные баллы XContest для качественного XC полёта

    Returns:
        Dict с метриками ячейки
    """
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
        return {
            "cell_id": f"{cell_lat}_{cell_lon}",
            "cell_lat": cell_lat,
            "cell_lon": cell_lon,
            "total_flights": 0,
            "flyable_days": 0,
            "unique_days": 0,
            "years_available": [],
            "avg_flights_per_flyable_day": 0.0,
        }

    # Фильтруем по качеству XC (баллы XContest)
    if min_xc_points > 0:
        df_cell = df_cell[df_cell["points"] >= min_xc_points].copy()
        if len(df_cell) == 0:
            return {
                "cell_id": f"{cell_lat}_{cell_lon}",
                "cell_lat": cell_lat,
                "cell_lon": cell_lon,
                "total_flights": 0,
                "flyable_days": 0,
                "unique_days": 0,
                "years_available": [],
                "avg_flights_per_flyable_day": 0.0,
            }

    # Агрегация по дням
    df_cell["date_only"] = pd.to_datetime(df_cell["date"]).dt.tz_localize(None).dt.normalize()
    daily_flights = df_cell.groupby("date_only").agg(flight_count=("id", "count")).reset_index()

    # После фильтрации по качеству: 1+ качественный полёт = летный день
    flyable_days = (daily_flights["flight_count"] >= 1).sum()
    total_flights = len(df_cell)
    unique_days = len(daily_flights)

    # Годы с данными
    years = sorted(df_cell["date"].dt.year.unique().tolist())

    # Средняя интенсивность в лётные дни
    flyable_subset = daily_flights[daily_flights["flight_count"] >= 1]
    avg_flights = flyable_subset["flight_count"].mean() if len(flyable_subset) > 0 else 0.0

    return {
        "cell_id": f"{cell_lat}_{cell_lon}",
        "cell_lat": cell_lat,
        "cell_lon": cell_lon,
        "total_flights": total_flights,
        "flyable_days": flyable_days,
        "unique_days": unique_days,
        "years_available": years,
        "avg_flights_per_flyable_day": avg_flights,
    }


def check_weather_coverage(
    cell_lat: int,
    cell_lon: int,
    cache: WeatherCache,
    years: Optional[List[int]] = None,
    season: Tuple[Tuple[int, int], Tuple[int, int]] = (SEASON_START, SEASON_END),
) -> float:
    """
    Проверяет покрытие погодными данными для ячейки.

    Returns:
        float: процент дней с данными (0-100) в сезонном окне
    """
    if years is None:
        years = [2021, 2022, 2023, 2024, 2025, 2026]
    (s_month, s_day), (e_month, e_day) = season
    target_dates = []
    for y in years:
        rng = pd.date_range(
            pd.Timestamp(year=y, month=s_month, day=s_day),
            pd.Timestamp(year=y, month=e_month, day=e_day),
        )
        target_dates.extend(rng)

    available_count = 0
    for date in target_dates:
        sample = cache.load_sample(cell_lat, cell_lon, date, 12)
        if sample:
            available_count += 1

    coverage = 100.0 * available_count / len(target_dates) if target_dates else 0.0
    return coverage


def get_cell_statistics(
    flights_dir: str = "data/flights",
    cache_root: str = "data/gfs/cache",
    output_path: str = "data/processed/cell_quality.csv",
    bbox: Optional[str] = None,
    min_xc_points: int = 10,
    flights_source: str = "world",
    flights_cache: Optional[str] = "data/processed/world_flights.pkl",
) -> pd.DataFrame:
    """
    Анализирует все доступные ячейки.

    Args:
        flights_dir: директория с файлами полётов
        cache_root: корневая директория кэша погоды
        output_path: путь для сохранения CSV
        bbox: ограничение области в формате 'lon_min,lat_min,lon_max,lat_max'.
              Если None, анализируются все ячейки.
        min_xc_points: минимальные баллы XContest для качественного XC полёта

    Returns:
        DataFrame с метриками для каждой ячейки
    """
    print("Загружаю все полёты...")
    if flights_source == "world":
        df_flights = load_world_flights(
            data_dir=flights_dir, bbox=bbox, cache_path=flights_cache
        )
    else:
        df_flights = load_flights_to_dataframe(data_dir=flights_dir)

    # Парсим bbox если указан
    lon_min, lat_min, lon_max, lat_max = None, None, None, None
    if bbox:
        try:
            lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))
            print(f"\nОграничение области: bbox=[{lon_min},{lat_min},{lon_max},{lat_max}]")
        except ValueError:
            print(
                f"\n⚠️ Неверный формат bbox: '{bbox}'. Ожидается 'lon_min,lat_min,lon_max,lat_max'."
            )
            print("Анализирую все ячейки.")

    # Получаем список уникальных ячеек из полётов
    df_geo = df_flights.dropna(subset=["takeoff_lat", "takeoff_lon"]).copy()
    df_geo["cell_lat"] = df_geo["takeoff_lat"].apply(lambda x: int(np.floor(x)))
    df_geo["cell_lon"] = df_geo["takeoff_lon"].apply(lambda x: int(np.floor(x)))

    # Фильтруем по bbox если указан
    if all(v is not None for v in [lon_min, lat_min, lon_max, lat_max]):
        df_geo = df_geo[
            (df_geo["takeoff_lon"] >= lon_min)
            & (df_geo["takeoff_lon"] <= lon_max)
            & (df_geo["takeoff_lat"] >= lat_min)
            & (df_geo["takeoff_lat"] <= lat_max)
        ].copy()
        print(f"Отфильтровано {len(df_geo)} полётов в указанной области")

    unique_cells = df_geo[["cell_lat", "cell_lon"]].drop_duplicates().values.tolist()
    print(f"Найдено {len(unique_cells)} уникальных ячеек с полётами")

    # Проверяем доступность погодных данных
    cache = WeatherCache(cache_root=cache_root)
    cache_cells_path = Path(cache_root) / "cells"
    available_weather_cells = []

    if cache_cells_path.exists():
        for cell_dir in cache_cells_path.iterdir():
            if cell_dir.is_dir() and "_" in cell_dir.name:
                try:
                    lat, lon = map(int, cell_dir.name.split("_"))
                    available_weather_cells.append((lat, lon))
                except ValueError:
                    continue

    print(f"Доступно {len(available_weather_cells)} ячеек с погодными данными")

    # Пересечение: ячейки с полётами И погодными данными
    cells_to_analyze = [
        (lat, lon) for lat, lon in unique_cells if (lat, lon) in available_weather_cells
    ]

    print(f"\nАнализирую {len(cells_to_analyze)} ячеек с полётами И погодой...\n")

    results = []
    for cell_lat, cell_lon in tqdm(cells_to_analyze, desc="Анализ ячеек"):
        # Анализ полётов
        flight_stats = analyze_cell_flights(df_flights, cell_lat, cell_lon, min_xc_points)

        # Проверка покрытия погодными данными
        weather_coverage = check_weather_coverage(cell_lat, cell_lon, cache)

        # Объединяем метрики
        result = {**flight_stats, "weather_coverage": weather_coverage}

        # Качественная оценка (чем выше, тем лучше)
        quality_score = (
            result["total_flights"] * 0.4
            + result["flyable_days"] * 10.0
            + result["weather_coverage"] * 2.0
        )
        result["quality_score"] = quality_score

        # Рекомендация
        if (
            result["total_flights"] >= 200
            and result["flyable_days"] >= 30
            and result["weather_coverage"] >= 80
        ):
            result["recommendation"] = "include"
        elif result["total_flights"] >= 100 and result["flyable_days"] >= 20:
            result["recommendation"] = "borderline"
        else:
            result["recommendation"] = "exclude"

        results.append(result)

    # Создаём DataFrame и сортируем по quality_score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("quality_score", ascending=False)

    # Сохраняем
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Результаты сохранены в {output_path}")

    # Печатаем статистику
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПО РЕКОМЕНДАЦИЯМ")
    print("=" * 60)
    print(df_results["recommendation"].value_counts())

    print("\n" + "=" * 60)
    print("ТОП-10 ЯЧЕЕК ПО КАЧЕСТВУ")
    print("=" * 60)
    print(
        df_results[
            [
                "cell_id",
                "total_flights",
                "flyable_days",
                "weather_coverage",
                "quality_score",
                "recommendation",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return df_results


def select_quality_cells(
    cell_quality_path: str = "data/processed/cell_quality.csv",
    min_flights: int = 200,
    min_flyable_days: int = 30,
    min_weather_coverage: float = 80.0,
    min_regions: int = 3,
) -> List[str]:
    """
    Возвращает список cell_id ячеек, прошедших фильтрацию.

    Args:
        cell_quality_path: путь к CSV с метриками
        min_flights: минимум полётов всего
        min_flyable_days: минимум лётных дней
        min_weather_coverage: минимум покрытия погодой (%)
        min_regions: минимальное рекомендуемое количество ячеек

    Returns:
        List[str]: список cell_id вида ['45_11', '46_11', ...]
    """
    df = pd.read_csv(cell_quality_path)

    filtered = df[
        (df["total_flights"] >= min_flights)
        & (df["flyable_days"] >= min_flyable_days)
        & (df["weather_coverage"] >= min_weather_coverage)
    ]

    selected_cells = filtered["cell_id"].tolist()

    print(f"\n✓ Выбрано {len(selected_cells)} ячеек:")
    for cell_id in selected_cells:
        row = filtered[filtered["cell_id"] == cell_id].iloc[0]
        print(
            f"  - {cell_id}: {row['total_flights']} полётов, "
            f"{row['flyable_days']} лётных дней, "
            f"{row['weather_coverage']:.1f}% покрытие"
        )

    # Предупреждение если ячеек меньше чем регионов
    if len(selected_cells) < min_regions:
        print(
            f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Выбрано {len(selected_cells)} ячеек, "
            f"но настроено {min_regions} регионов."
        )
        print("   Рекомендуется увеличить область (--bbox) или снизить пороги фильтрации.")

    return selected_cells


if __name__ == "__main__":
    # Анализируем все ячейки
    df_quality = get_cell_statistics()

    # Выбираем качественные ячейки с консервативными порогами
    selected = select_quality_cells(min_flights=400, min_flyable_days=60)

    # Сохраняем список выбранных ячеек для dataset_builder_v2
    output_json = "data/processed/selected_cells.json"
    with open(output_json, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\n✓ Список выбранных ячеек сохранён в {output_json}")
