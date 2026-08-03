"""
Анализ качества ячеек и ОТБОР ячеек для обучения.

Отбор идёт по одним полётам — погодного кэша для новой ячейки ещё не существует,
и требовать его значило бы уметь отбирать только то, что уже извлечено. Ячейка
проходит, если в сезонном окне с SELECT_FROM_YEAR набралось достаточно:

- quality_days: дней хотя бы с одним качественным полётом (points >= min_xc_points)
- good_days:    дней с полётом >= GOOD_DISTANCE_KM — метка модели вырождена там,
                где полёт такой длины почти не случается, сколько бы ни было взлётов
- sites:        разных стартов — одна гора это не ячейка, а одна гора

Пороги заданы НА ГРАДУС ШИРОТЫ и умножаются на LAT_STEP: они считают дни, а не
плотность, поэтому при уменьшении ячейки планка иначе поднялась бы сама собой, без
всякой связи с качеством данных. На сетке 1° это ровно те 150/60, которыми отобран
прод; на 0.75 — 112/45.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from paraglideml.data.dataset_builder import SEASON_END, SEASON_START
from paraglideml.data.flight_parsing import load_flights_to_dataframe, load_world_flights
from paraglideml.data.weather_cache import WeatherCache
from paraglideml.grid import LAT_STEP, cell_bounds
from paraglideml.grid import cell_id as make_cell_id

# Критерии отбора, на градус широты (см. модульную строку документации).
QUALITY_DAYS_PER_DEG = 150.0
GOOD_DAYS_PER_DEG = 60.0
MIN_SITES = 3  # НЕ масштабируется: это порог разнообразия, а не объёма
GOOD_DISTANCE_KM = 50.0
SELECT_FROM_YEAR = 2021

MIN_QUALITY_DAYS = int(round(QUALITY_DAYS_PER_DEG * LAT_STEP))
MIN_GOOD_DAYS = int(round(GOOD_DAYS_PER_DEG * LAT_STEP))


def get_cell_from_coords(lat: float, lon: float) -> Tuple[float, float]:
    """Якорь (юго-западный угол) ячейки, содержащей точку."""
    lat0, lon0, _, _ = cell_bounds(make_cell_id(lat, lon))
    return lat0, lon0


def cell_flight_stats(
    df_flights: pd.DataFrame,
    season: Tuple[Tuple[int, int], Tuple[int, int]] = (SEASON_START, SEASON_END),
    from_year: int = SELECT_FROM_YEAR,
    min_xc_points: int = 10,
) -> pd.DataFrame:
    """
    Метрики отбора по каждой ячейке, где вообще были полёты.

    Одна группировка по всему кадру вместо прохода по ячейкам: на мировом экспорте
    (5M полётов) поячеечная фильтрация — это часы, группировка — секунды.
    """
    df = df_flights.dropna(subset=["takeoff_lat", "takeoff_lon"]).copy()
    df["day"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    (s_month, _), (e_month, _) = season
    df = df[(df["day"].dt.year >= from_year) & (df["day"].dt.month.between(s_month, e_month))]
    if df.empty:
        return pd.DataFrame()

    df["cell_id"] = [
        make_cell_id(la, lo) for la, lo in zip(df["takeoff_lat"], df["takeoff_lon"])
    ]

    site_col = "takeoff_name" if "takeoff_name" in df.columns else "takeoff_lat"
    q = df[df["points"] >= min_xc_points] if min_xc_points > 0 else df
    if q.empty:
        return pd.DataFrame()

    g = q.groupby("cell_id").agg(
        total_flights=("day", "size"),
        quality_days=("day", "nunique"),
        sites=(site_col, "nunique"),
        pilots=("pilot_id", "nunique"),
    )
    good = q[q["distance"] >= GOOD_DISTANCE_KM].groupby("cell_id")["day"].nunique()
    g["good_days"] = good.reindex(g.index).fillna(0).astype(int)
    g["years"] = q.groupby("cell_id")["day"].agg(lambda s: s.dt.year.nunique()).reindex(g.index)

    g = g.reset_index()
    anchors = [cell_bounds(c) for c in g["cell_id"]]
    g["cell_lat"] = [a[0] for a in anchors]
    g["cell_lon"] = [a[1] for a in anchors]
    g["passes"] = (
        (g["quality_days"] >= MIN_QUALITY_DAYS)
        & (g["good_days"] >= MIN_GOOD_DAYS)
        & (g["sites"] >= MIN_SITES)
    )
    return g.sort_values("good_days", ascending=False).reset_index(drop=True)


def check_weather_coverage(
    cell_lat: float,
    cell_lon: float,
    cache: WeatherCache,
    years: Optional[List[int]] = None,
    season: Tuple[Tuple[int, int], Tuple[int, int]] = (SEASON_START, SEASON_END),
) -> float:
    """Процент дней сезонного окна, для которых у ячейки есть срез 12 UTC."""
    if years is None:
        years = [2021, 2022, 2023, 2024, 2025, 2026]
    (s_month, s_day), (e_month, e_day) = season
    target_dates: List[pd.Timestamp] = []
    for y in years:
        target_dates.extend(
            pd.date_range(
                pd.Timestamp(year=y, month=s_month, day=s_day),
                pd.Timestamp(year=y, month=e_month, day=e_day),
            )
        )
    if not target_dates:
        return 0.0
    available = sum(1 for d in target_dates if cache.load_sample(cell_lat, cell_lon, d, 12))
    return 100.0 * available / len(target_dates)


def get_cell_statistics(
    flights_dir: str = "data/flights",
    cache_root: str = "data/gfs/cache",
    output_path: str = "data/processed/cell_quality.csv",
    bbox: Optional[str] = None,
    min_xc_points: int = 10,
    flights_source: str = "world",
    flights_cache: Optional[str] = "data/processed/world_flights.pkl",
    years: Optional[Sequence[int]] = None,
    with_weather: bool = False,
) -> pd.DataFrame:
    """
    Считает метрики отбора по всем ячейкам с полётами и пишет CSV.

    `bbox` ('lon_min,lat_min,lon_max,lat_max') — операционный охват продукта: то, для
    чего мы готовы извлекать GFS. Это единственная география, которую здесь можно
    задавать; накладывать поверх неё ещё и широтный пояс нельзя — так уже терялись
    ячейки, проходившие критерии (Babadağ, Algodonales).

    `with_weather` дополнительно меряет покрытие погодным кэшем у прошедших ячеек —
    диагностика, на отбор не влияет (кэша для новой сетки ещё нет).
    """
    print("Загружаю полёты...")
    if flights_source == "world":
        df_flights = load_world_flights(
            data_dir=flights_dir,
            bbox=bbox,
            years=list(years) if years else None,
            cache_path=flights_cache,
        )
    else:
        df_flights = load_flights_to_dataframe(data_dir=flights_dir)

    if bbox:
        lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox.split(","))
        print(f"Охват: bbox=[{lon_min},{lat_min},{lon_max},{lat_max}]")
        df_flights = df_flights[
            df_flights["takeoff_lon"].between(lon_min, lon_max)
            & df_flights["takeoff_lat"].between(lat_min, lat_max)
        ]

    df_results = cell_flight_stats(df_flights, min_xc_points=min_xc_points)
    if df_results.empty:
        print("Ни одной ячейки с полётами в охвате.")
        return df_results

    if with_weather:
        cache = WeatherCache(cache_root=cache_root)
        passing = df_results[df_results["passes"]]
        print(f"Меряю покрытие погодой у {len(passing)} прошедших ячеек...")
        cov = {
            r.cell_id: check_weather_coverage(r.cell_lat, r.cell_lon, cache)
            for r in passing.itertuples()
        }
        df_results["weather_coverage"] = df_results["cell_id"].map(cov)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Метрики по {len(df_results)} ячейкам -> {output_path}")

    n_pass = int(df_results["passes"].sum())
    print(
        f"\nКритерии (сетка {LAT_STEP:g}°): quality_days>={MIN_QUALITY_DAYS}, "
        f"good_days>={MIN_GOOD_DAYS}, sites>={MIN_SITES}\n"
        f"Проходят: {n_pass} из {len(df_results)}"
    )
    cols = ["cell_id", "total_flights", "quality_days", "good_days", "sites", "pilots"]
    print("\nТОП-10 по good_days:")
    print(df_results[cols].head(10).to_string(index=False))
    return df_results


def select_quality_cells(
    cell_quality_path: str = "data/processed/cell_quality.csv",
    min_quality_days: int = MIN_QUALITY_DAYS,
    min_good_days: int = MIN_GOOD_DAYS,
    min_sites: int = MIN_SITES,
    verbose: bool = True,
) -> List[str]:
    """Список cell_id, прошедших пороги, из CSV, посчитанного get_cell_statistics."""
    df = pd.read_csv(cell_quality_path)
    filtered = df[
        (df["quality_days"] >= min_quality_days)
        & (df["good_days"] >= min_good_days)
        & (df["sites"] >= min_sites)
    ]
    selected = filtered["cell_id"].astype(str).tolist()

    if verbose:
        print(f"\n✓ Отобрано {len(selected)} ячеек:")
        for r in filtered.itertuples():
            print(
                f"  - {r.cell_id}: {r.total_flights} полётов, "
                f"{r.quality_days} качественных дней, "
                f"{r.good_days} дней >={GOOD_DISTANCE_KM:.0f} км, {r.sites} стартов"
            )
    # Сортируем географически: так диффы selected_cells.json читаемы.
    return sorted(selected, key=lambda c: (cell_bounds(c)[0], cell_bounds(c)[1]))


def cell_stats_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Короткая сводка для лога/тестов."""
    return {
        "cells": len(df),
        "passing": int(df["passes"].sum()) if "passes" in df else 0,
    }


if __name__ == "__main__":
    df_quality = get_cell_statistics()
    selected = select_quality_cells()
    output_json = "data/processed/selected_cells.json"
    with open(output_json, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\n✓ Список выбранных ячеек сохранён в {output_json}")
