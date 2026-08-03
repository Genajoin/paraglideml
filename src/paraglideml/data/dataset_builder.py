"""
Мульти-ячеечный dataset builder для обучения на нескольких локациях.

Создаёт объединённый датасет из качественных ячеек с:
- Погодными фичами
- Метками is_flyable
- Confidence-weighted labels для учёта неопределённости
"""

import json
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from paraglideml.data.flight_parsing import load_flights_to_dataframe, load_world_flights
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


FORECAST_HOURS = (6, 12, 18)

# Training season window, as (start month-day, end month-day). The historical
# default was 15 May - 15 Sep, which matched the weather cache rather than the
# flying: in the Alps cells April alone yields more >=50 km flights than July,
# and March more than September. Widened to 1 Mar - 31 Oct once the GFS cache
# covers it; override per call for a cell whose season sits elsewhere.
SEASON_START = (3, 1)
SEASON_END = (10, 31)


def compute_day_features(
    cache: WeatherCache, cell_lat: int, cell_lon: int, date: pd.Timestamp
) -> Optional[dict]:
    """
    Build the full feature record for one cell-day.

    Loads the 06/12/18 UTC slices, uses 12 UTC as the anchor (full feature vector)
    and appends the diurnal aggregates. Returns None if the 12 UTC anchor is
    missing. Shared by the dataset builder (training) and the forecast/inference
    path so both compute features identically.
    """
    samples = {h: cache.load_sample(cell_lat, cell_lon, date, h) for h in FORECAST_HOURS}
    base = samples.get(12)
    if not base:
        return None

    feats_by_hour = {h: s["features"] for h, s in samples.items() if s}

    def _vals(key):
        return [f[key] for f in feats_by_hour.values() if key in f]

    record = dict(base["features"])

    # Дневные максимумы опасного ветра/порывов и пиковой неустойчивости/CAPE
    for src, dst in [
        ("cape", "cape_daymax"),
        ("wind_speed_850", "ws850_daymax"),
        ("wind_speed_700", "ws700_daymax"),
        ("gust_10m", "gust_daymax"),
        ("lapse_low_mean", "lapse_low_daymax"),
        # Peak convective / storm potential reached during the day
        ("k_index", "k_index_daymax"),
        ("total_totals", "total_totals_daymax"),
        ("cape_shear", "cape_shear_daymax"),
    ]:
        vals = _vals(src)
        record[dst] = max(vals) if vals else float(record.get(src, 0.0))

    # Прирост CAPE с утра к полудню (накопление дневного прогрева)
    cape_06 = feats_by_hour.get(6, {}).get("cape")
    cape_12 = feats_by_hour.get(12, {}).get("cape")
    record["cape_amp"] = (
        (cape_12 - cape_06) if (cape_06 is not None and cape_12 is not None) else 0.0
    )

    # Сильнейший восходящий поток за день (наиболее отрицательная омега в нижнем слое)
    omega_vals = _vals("omega_low_mean")
    record["omega_low_min"] = (
        min(omega_vals) if omega_vals else float(record.get("omega_low_mean", 0.0))
    )

    return record


def build_cell_dataset(
    cell_id: str,
    df_flights: pd.DataFrame,
    cache: WeatherCache,
    min_xc_points: int = 10,
    cell_terrain: Optional[dict] = None,
    available_years: Optional[List[int]] = None,
    data_max_date: Optional[pd.Timestamp] = None,
    season: Tuple[Tuple[int, int], Tuple[int, int]] = (SEASON_START, SEASON_END),
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

    # Агрегация по дням. Кроме счётчика полётов сохраняем дневные агрегаты
    # дистанции/очков XContest — это сырьё для distance-цели «хороший XC-день»
    # (см. target-methodology). Эти колонки НЕ используются как фичи модели
    # (они — исход, не предиктор) и исключаются в multiregional.drop_cols.
    df_cell["date_only"] = pd.to_datetime(df_cell["date"]).dt.tz_localize(None).dt.normalize()
    daily_flights = (
        df_cell.groupby("date_only")
        .agg(
            flight_count=("id", "count"),
            dist_max=("distance", "max"),
            dist_mean=("distance", "mean"),
            dist_sum=("distance", "sum"),
            pts_max=("points", "max"),
        )
        .reset_index()
    )
    daily_flights.rename(columns={"date_only": "date"}, inplace=True)

    # Создаём target timeline по сезонному окну (по умолчанию 1 марта - 31 октября).
    # Для текущего/частичного года обрезаем конец по последней дате данных — иначе
    # будущие дни без полётов стали бы ложными «нелётными» нулями.
    if available_years is None:
        available_years = [2021, 2022, 2023, 2024, 2025]
    (s_month, s_day), (e_month, e_day) = season
    target_dates = []
    for y in available_years:
        season_start = pd.Timestamp(year=y, month=s_month, day=s_day)
        season_end = pd.Timestamp(year=y, month=e_month, day=e_day)
        if data_max_date is not None:
            season_end = min(season_end, pd.Timestamp(data_max_date))
        if season_start > season_end:
            continue
        target_dates.extend(pd.date_range(season_start, season_end))

    df_target = pd.DataFrame({"date": target_dates})
    df_target["date"] = pd.to_datetime(df_target["date"]).dt.normalize()
    df_target = df_target.merge(daily_flights, on="date", how="left").fillna(0)
    # После фильтрации по качеству: 1+ качественный полёт = летный день
    df_target["is_flyable"] = (df_target["flight_count"] >= 1).astype(int)

    # Добавляем day_of_week для вычисления is_weekend
    df_target["is_weekend"] = (df_target["date"].dt.dayofweek >= 5).astype(int)

    # Вычисляем confidence для каждой строки
    df_target["label_confidence"] = df_target.apply(compute_label_confidence, axis=1)

    # Мержим с погодными данными: для каждого дня берём 06/12/18 UTC и собираем
    # фичи через общую compute_day_features (та же логика в инференсе/forecast).
    weather_records = []
    for date in df_target["date"]:
        record = compute_day_features(cache, cell_lat, cell_lon, date)
        if record is None:
            continue  # нет опорного среза 12 UTC
        record = dict(record)
        record["date"] = date
        weather_records.append(record)

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

    # Спот-центричный террейн (фичи модели, не исход): известные высота/горность из
    # cell_terrain.json (`paraglideml data terrain`). Slope-wind НЕ добавляем —
    # на GFS-ветре прирост в пределах шума (см. terrain.slope_wind_alignment).
    if cell_terrain and cell_id in cell_terrain:
        tc = cell_terrain[cell_id]
        dataset["elevation"] = float(tc.get("elevation", 0.0))
        dataset["mountainess"] = float(tc.get("mountainess", 0.0))

    return dataset


def build_multicell_dataset(
    cells: Optional[List[str]] = None,
    flights_dir: str = "data/flights",
    cache_root: str = "data/gfs/cache",
    selected_cells_path: str = "data/processed/selected_cells.json",
    output_path: str = "data/processed/multicell_dataset.csv",
    min_xc_points: int = 10,
    flights_source: str = "world",
    flights_cache: Optional[str] = "data/processed/world_flights.pkl",
    season: Tuple[Tuple[int, int], Tuple[int, int]] = (SEASON_START, SEASON_END),
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
        flights_source: 'world' — мировой экспорт 2006-2026 (полный год, координаты
                        на каждый полёт); 'legacy' — старые xcontest_flights_*.json
        flights_cache: pickle-кэш разобранного мирового экспорта (None — не кэшировать)
        season: сезонное окно ((месяц, день) начала, (месяц, день) конца)

    Returns:
        pd.DataFrame с объединённым датасетом
    """
    # Загружаем список ячеек
    if cells is None:
        with open(selected_cells_path, "r") as f:
            cells = json.load(f)

    print("Загружаю полёты...")
    if flights_source == "world":
        # Ограничиваем разбор bbox'ом выбранных ячеек: мировой экспорт — 5M полётов
        # в 15.5k файлах, а нужна лишь горстка ячеек.
        lats = [int(c.split("_")[0]) for c in cells]
        lons = [int(c.split("_")[1]) for c in cells]
        bbox = f"{min(lons)},{min(lats)},{max(lons) + 1},{max(lats) + 1}"
        df_flights = load_world_flights(data_dir=flights_dir, bbox=bbox, cache_path=flights_cache)
    else:
        df_flights = load_flights_to_dataframe(data_dir=flights_dir)

    print("Инициализирую weather cache...")
    cache = WeatherCache(cache_root=cache_root)

    # Спот-центричный террейн (если построен) — добавит фичи elevation/mountainess.
    from .terrain import load_cell_terrain

    cell_terrain = load_cell_terrain()
    if cell_terrain:
        print(f"Террейн загружен для {len(cell_terrain)} ячеек (elevation/mountainess).")
    else:
        print("Террейн не найден (cell_terrain.json) — фичи рельефа пропущены.")

    # Годы и последняя дата данных определяются автоматически из полётов: так
    # подключение нового сезона (напр. частичного 2026) не требует правок кода —
    # таймлайн обрежется по факту (см. build_cell_dataset).
    available_years = sorted(int(y) for y in df_flights["year"].dropna().unique())
    data_max_date = pd.to_datetime(df_flights["date"]).dt.tz_localize(None).max().normalize()
    print(f"Годы в данных: {available_years}; последняя дата: {data_max_date.date()}")

    print(f"\nСоздаю датасеты для {len(cells)} ячеек с фильтром XC>={min_xc_points} очков...\n")

    all_datasets = []
    for cell_id in tqdm(cells, desc="Обработка ячеек"):
        cell_df = build_cell_dataset(
            cell_id, df_flights, cache, min_xc_points, cell_terrain,
            available_years=available_years, data_max_date=data_max_date, season=season,
        )
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
