from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygrib
from tqdm import tqdm

from ..config import GFS_ANL_DIR, GFS_CACHE_DIR

# =============================================================================
# SCHEMA CONFIGURATION
# =============================================================================

PRESSURE_LEVELS = [
    1000,
    975,
    950,
    925,
    900,
    850,
    800,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
    200,
]

EXTRACT_LIST = [
    ("Temperature", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("Geopotential height", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("Relative humidity", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("U component of wind", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("V component of wind", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("Vertical velocity", "isobaricInhPa", PRESSURE_LEVELS, "hPa"),
    ("2 metre temperature", "heightAboveGround", [2], "m"),
    ("2 metre dewpoint temperature", "heightAboveGround", [2], "m"),
    ("10 metre U wind component", "heightAboveGround", [10], "m"),
    ("10 metre V wind component", "heightAboveGround", [10], "m"),
    ("Convective available potential energy", "surface", [0], "sfc"),
    ("Convective inhibition", "surface", [0], "sfc"),
    ("Surface pressure", "surface", [0], "sfc"),
    ("Wind speed (gust)", "surface", [0], "sfc"),
    ("Total Cloud Cover", "atmosphere", [0], "atm"),
    ("Visibility", "surface", [0], "sfc"),
]

# =============================================================================
# HELPERS
# =============================================================================


def parse_date_ranges(date_ranges_str: str) -> List[datetime]:
    datetimes = []
    ranges = date_ranges_str.split(",")
    for r in ranges:
        if ":" in r:
            start_str, end_str = r.split(":")
            start = datetime.strptime(start_str.strip(), "%Y-%m-%d")
            end = datetime.strptime(end_str.strip(), "%Y-%m-%d")
            curr = start
            while curr <= end:
                for hour in [6, 12, 18]:
                    datetimes.append(curr.replace(hour=hour))
                curr += timedelta(days=1)
    return sorted(list(set(datetimes)))


def parse_bbox(bbox_str: str) -> List[Tuple[int, int]]:
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    lon_min, lat_min, lon_max, lat_max = parts
    cells = []
    for lat in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            cells.append((lat, lon))
    return cells


def get_cache_path(base_dir: Path, lat: int, lon: int, dt: datetime) -> Path:
    return (
        base_dir
        / f"cells/{lat}_{lon}"
        / dt.strftime("%Y/%m")
        / dt.strftime("gfsanl_3_%Y%m%d_%H00_000.npz")
    )


# =============================================================================
# PYGRIB EXTRACTION
# =============================================================================


def process_grib_pygrib(
    grib_path: Path, cells: List[Tuple[int, int]]
) -> Optional[Dict[Tuple[int, int], Dict[str, Any]]]:
    results = {cell: {"values": [], "keys": []} for cell in cells}

    try:
        grbs = pygrib.open(str(grib_path))
    except Exception as e:
        print(f"Error opening {grib_path}: {e}")
        return None

    try:
        first_grb = grbs.message(1)
        lats = first_grb.distinctLatitudes
        lons = first_grb.distinctLongitudes

        def find_closest(val, vect):
            return int((np.abs(vect - val)).argmin())

        cell_indices = []
        for lat, lon in cells:
            lat_idx = find_closest(lat, lats)
            lon_idx = find_closest(lon, lons)
            cell_indices.append((lat_idx, lon_idx))
    except Exception as e:
        print(f"Error reading grid structure from {grib_path}: {e}")
        grbs.close()
        return None

    for grb in grbs:
        name = grb.name
        short_name = grb.shortName
        level = grb.level
        type_level = grb.typeOfLevel

        matched = False
        suffix = ""
        for target_name, target_type, target_levels, target_suffix in EXTRACT_LIST:
            if target_name in name and type_level == target_type and level in target_levels:
                matched = True
                suffix = f"{level}{target_suffix}"
                break

        if matched:
            key_name = f"{short_name}{suffix}"
            try:
                vals_array = grb.values
                for i, cell in enumerate(cells):
                    lat_idx, lon_idx = cell_indices[i]
                    val = vals_array[lat_idx, lon_idx]
                    results[cell]["values"].append(float(val))
                    results[cell]["keys"].append(key_name)
            except Exception as e:
                print(f"Error extracting values for {key_name}: {e}")

    grbs.close()
    return results


def run_gfs_cache_creation(
    dates: str,
    bbox: str,
    source_dir: Path = GFS_ANL_DIR,
    output_dir: Path = GFS_CACHE_DIR,
    force: bool = False,
):
    datetimes = parse_date_ranges(dates)
    cells = parse_bbox(bbox)

    print(f"Processing GFS cache for {len(datetimes)} timestamps and {len(cells)} cells.")

    stats = {"processed": 0, "skipped": 0, "missing": 0, "errors": 0}

    for dt in tqdm(datetimes, desc="Processing GRIBs"):
        needed_cells = []
        for cell in cells:
            p = get_cache_path(output_dir, cell[0], cell[1], dt)
            if force or not p.exists():
                needed_cells.append(cell)

        if not needed_cells:
            stats["skipped"] += 1
            continue

        grib_path = source_dir / dt.strftime("%Y-%m") / dt.strftime("gfsanl_3_%Y%m%d_%H00_000.grb2")
        if not grib_path.exists():
            grib_path_flat = source_dir / dt.strftime("gfsanl_3_%Y%m%d_%H00_000.grb2")
            if grib_path_flat.exists():
                grib_path = grib_path_flat
            else:
                stats["missing"] += 1
                continue

        try:
            data_map = process_grib_pygrib(grib_path, needed_cells)
            if not data_map:
                stats["errors"] += 1
                continue

            for (lat, lon), content in data_map.items():
                if not content["values"]:
                    continue

                out_path = get_cache_path(output_dir, lat, lon, dt)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    out_path,
                    values=np.array(content["values"], dtype=np.float32),
                    keys=np.array(content["keys"]),
                    lat=lat,
                    lon=lon,
                    timestamp=dt.isoformat(),
                )
            stats["processed"] += 1
        except Exception as e:
            print(f"Error processing {dt}: {e}")
            stats["errors"] += 1

    print(
        f"\nSummary: Processed={stats['processed']}, Skipped={stats['skipped']}, Missing={stats['missing']}, Errors={stats['errors']}"
    )
