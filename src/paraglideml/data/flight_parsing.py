import glob
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import pandas as pd

# Regex to extract coordinates from XContest takeoff link
# Example: ...filter[point]=13.779207 46.181068...
COORD_REGEX = re.compile(r"filter\[point\]=([\d\.]+)\s+([\d\.]+)")

# --- World XContest export (2006-2026) -------------------------------------
# Layout: <root>/{YYYY}/{CC}-{YYYY}-{MM}.json, each a flat list of flights.
# Schema differs from the legacy per-country files: coordinates come straight
# from takeoff.lat/lon (per-flight launch point, ~0.2 km spread within a site,
# not the site-catalogue point behind takeoff.link), duration is in seconds,
# and there is no glider / flight id / countries list.
WORLD_FLIGHTS_DIR = "2006-2026.flights/extract"

# Records the source itself marks as broken or that are physically impossible.
# Kept deliberately loose: this only drops garbage, never marginal flights.
MAX_PLAUSIBLE_DISTANCE_KM = 500.0
MAX_PLAUSIBLE_SPEED_KMH = 150.0


def parse_flight(flight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single flight dictionary into a flat structure.
    """
    flat_flight = {
        "id": flight.get("id"),
        "pilot_id": flight.get("pilot", {}).get("id"),
        "pilot_name": flight.get("pilot", {}).get("name"),
        "date": flight.get("pointStart", {}).get("time"),
        "distance": flight.get("league", {}).get("route", {}).get("distance"),
        "points": flight.get("league", {}).get("route", {}).get("points"),
        "glider": flight.get("glider", {}).get("name"),
        "glider_class": flight.get("glider", {}).get("class"),
        "takeoff_name": flight.get("takeoff", {}).get("name"),
        "country": flight.get("takeoff", {}).get("countryIso"),
    }

    # Extract coordinates
    takeoff_link = flight.get("takeoff", {}).get("link", "")
    if takeoff_link:
        match = COORD_REGEX.search(takeoff_link)
        if match:
            # XContest usually provides LON LAT
            flat_flight["takeoff_lon"] = float(match.group(1))
            flat_flight["takeoff_lat"] = float(match.group(2))
        else:
            flat_flight["takeoff_lon"] = None
            flat_flight["takeoff_lat"] = None
    else:
        flat_flight["takeoff_lon"] = None
        flat_flight["takeoff_lat"] = None

    return flat_flight


def load_flights_to_dataframe(
    data_dir: str = "data/flights", pattern: str = "*.json"
) -> pd.DataFrame:
    """
    Loads all JSON flight files from the directory matching the pattern into a Pandas DataFrame.
    """
    files = glob.glob(str(Path(data_dir) / pattern))
    all_flights = []

    print(f"Found {len(files)} files in {data_dir} matching {pattern}")

    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Data is expected to be a list of flights
                if isinstance(data, list):
                    for flight in data:
                        all_flights.append(parse_flight(flight))
                else:
                    print(f"Warning: File {file_path} does not contain a list.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    df = pd.DataFrame(all_flights)

    # Post-processing
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

    return df


def _parse_world_file(args: Tuple[str, Optional[Tuple[float, float, float, float]]]) -> list:
    """Read one {CC}-{YYYY}-{MM}.json partition into raw tuples (worker function)."""
    file_path, bbox = args
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # a single unreadable partition must not kill the load
        print(f"Error loading {file_path}: {e}")
        return rows

    for r in data:
        t = r.get("takeoff") or {}
        lat, lon = t.get("lat"), t.get("lon")
        if lat is None or lon is None:
            continue
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
        route = r.get("route") or {}
        pilot = r.get("pilot") or {}
        rows.append(
            (
                r.get("startTime"),
                lat,
                lon,
                t.get("name"),
                t.get("countryIso"),
                route.get("type"),
                route.get("distance"),
                route.get("points"),
                route.get("avgSpeed"),
                r.get("duration"),
                pilot.get("id"),
                pilot.get("name"),
            )
        )
    return rows


def load_world_flights(
    data_dir: str = "data/flights",
    world_subdir: str = WORLD_FLIGHTS_DIR,
    bbox: Optional[str] = None,
    years: Optional[Sequence[int]] = None,
    cache_path: Optional[str] = None,
    workers: Optional[int] = None,
    clean: bool = True,
) -> pd.DataFrame:
    """
    Load the world XContest export (2006-2026) into the same column contract that
    `load_flights_to_dataframe` returns, so the cell analyzer / dataset builder can
    consume either source unchanged.

    The export is ~5.0M flights across 15.5k partition files; parsing all of it takes
    a couple of minutes, so pass `bbox` (as "lon_min,lat_min,lon_max,lat_max") and/or
    `years` to cut the work, and `cache_path` to memoise the parsed frame as a pickle.

    `clean=True` drops the source's garbage: exact duplicates (same pilot + start time
    + distance, which the export carries for ~0.25% of rows), zero/absurd distances and
    impossible ground speeds. It never filters on XC quality — that stays the caller's
    job via `min_xc_points`.

    No `glider` / `glider_class`: the export has no glider block. Nothing reads those
    columns today (the legacy loader parses them, but no analyzer, builder or feature
    ever touches them), so they are simply absent rather than present-and-always-None.
    """
    # The cache key has to carry the filters, or a later call with a wider bbox /
    # year set would silently be served the narrower frame parsed earlier.
    if cache_path:
        scope = f"{bbox or 'world'}|{','.join(str(y) for y in years) if years else 'all'}"
        digest = hashlib.sha1(scope.encode()).hexdigest()[:10]
        p = Path(cache_path)
        cache_path = str(p.with_name(f"{p.stem}.{digest}{p.suffix}"))
        if Path(cache_path).exists():
            return cast(pd.DataFrame, pd.read_pickle(cache_path))

    root = Path(data_dir) / world_subdir
    if not root.exists():
        raise FileNotFoundError(f"World flights export not found: {root}")

    files = sorted(glob.glob(str(root / "*" / "*.json")))
    if years is not None:
        wanted = {str(y) for y in years}
        files = [f for f in files if Path(f).parent.name in wanted]
    if not files:
        raise FileNotFoundError(f"No partition files under {root} (years={years})")

    bbox_t: Optional[Tuple[float, float, float, float]] = None
    if bbox:
        lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox.split(","))
        bbox_t = (lon_min, lat_min, lon_max, lat_max)

    print(f"Загружаю мировой экспорт: {len(files)} партиций из {root}")
    workers = workers or min(8, (os.cpu_count() or 2))
    rows: list = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(_parse_world_file, ((f, bbox_t) for f in files), chunksize=16):
            rows.extend(chunk)

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "takeoff_lat",
            "takeoff_lon",
            "takeoff_name",
            "country",
            "route_type",
            "distance",
            "points",
            "avg_speed",
            "duration",
            "pilot_id",
            "pilot_name",
        ],
    )
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)
    for c in ("takeoff_lat", "takeoff_lon", "distance", "points", "avg_speed"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if clean:
        before = len(df)
        keep = (
            ~df.duplicated(["pilot_id", "date", "distance"])
            & (df["distance"] > 0)
            & (df["distance"] < MAX_PLAUSIBLE_DISTANCE_KM)
            & (df["avg_speed"] < MAX_PLAUSIBLE_SPEED_KMH)
        )
        df = cast(pd.DataFrame, df.loc[keep])
        print(f"  отфильтровано {before - len(df)} мусорных записей из {before}")

    # Synthetic stable id: the export has no flight id, and the aggregation path
    # counts flights via this column.
    df = df.reset_index(drop=True)
    df["id"] = df["pilot_id"].astype("int64").astype(str) + ":" + df["date"].astype("int64").astype(str)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df


if __name__ == "__main__":
    # Test run
    df = load_flights_to_dataframe()
    print(df.head())
    print(df.info())
