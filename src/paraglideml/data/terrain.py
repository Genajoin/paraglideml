"""
Spot-centric terrain & slope-orientation features from FlyBeeper spot files.

Launch-site data is *known*, so we don't sample a DEM raster: FlyBeeper's
``dhv_loc.geojson`` gives per-site altitude and flying directions, and
``takeoff.geojson`` gives 8-way launch-orientation flags. Per cell we aggregate:

  elevation      median altitude of the cell's launch sites (real, not cell-mean)
  mountainess    altitude spread (p90-p10)/1000, clamped — relief proxy from sites
  orientations   8-vector: fraction of sites launchable at N,NE,E,SE,S,SW,W,NW

The orientations unlock a physical, *dynamic* flyability feature: given the forecast
wind, is there a launch facing into it? `slope_wind_alignment` scores how well the
low-level wind blows onto an available slope — the exact thing that decides whether a
windy day is flyable only off certain aspects (or not at all).

`data terrain` writes the small data/processed/cell_terrain.json that the dataset
builder and forecast path consume; the source geojson is read only here.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import CELL_TERRAIN_PATH, FLYBEEPER_SITES_DIR, PROCESSED_DATA_DIR
from ..grid import cell_id as _cell_of

# Orientation order: index k -> compass bearing k*45 (N, NE, E, SE, S, SW, W, NW).
ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
# dhv directionsText tokens (German/intl) -> orientation index.
_DIR_TOKEN = {"N": 0, "NO": 1, "NE": 1, "O": 2, "E": 2, "SO": 3, "SE": 3,
              "S": 4, "SW": 5, "W": 6, "NW": 7}


def _takeoff_orientations(props: dict) -> set:
    """Orientation indices flagged launchable in a takeoff.geojson feature."""
    out = set()
    for k, bearing in enumerate(ORIENTATIONS):
        if str(props.get(f"{bearing:03d}_Launch", "0")) == "1":
            out.add(k)
    return out


def _dhv_orientations(directions_text: Optional[str]) -> set:
    """Orientation indices parsed from a dhv directionsText like 'O, W'."""
    out = set()
    if not directions_text:
        return out
    for tok in str(directions_text).replace(";", ",").split(","):
        idx = _DIR_TOKEN.get(tok.strip().upper())
        if idx is not None:
            out.add(idx)
    return out


def slope_wind_alignment(u: float, v: float, orientations: Optional[List[float]]):
    """
    Score how well the wind blows onto an available launch slope.

    NOTE: empirically dormant. Adding this as a model feature gave only +0.003 AP
    over known heights on the good_xc target — GFS surface wind is too coarse and
    many launches are sheltered by terrain behind them, so the score is noisy.
    Kept for a future, finer wind source (e.g. higher-res model or ridge-level u/v).

    A slope facing bearing theta is soarable when the wind comes FROM theta. With
    u,v (eastward/northward m/s) the meteorological from-direction is
    atan2(-u,-v). Returns (alignment, aligned_speed):
      alignment    max over orientations of cos(from_dir - theta) weighted by the
                   fraction of sites offering that aspect (in [0,1])
      aligned_speed alignment * wind speed (a soarable-into-slope wind magnitude)
    """
    if not orientations:
        return 0.0, 0.0
    speed = math.hypot(u, v)
    if speed < 1e-6:
        return 0.0, 0.0
    from_dir = math.degrees(math.atan2(-u, -v)) % 360.0
    best = 0.0
    for k, frac in enumerate(orientations):
        if frac <= 0:
            continue
        align = math.cos(math.radians(from_dir - ORIENTATIONS[k]))
        best = max(best, align * frac)
    best = max(0.0, best)
    return best, best * speed


def build_cell_terrain(
    selected_cells_path: Optional[str] = None,
    sites_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Aggregate FlyBeeper launch sites into per-cell terrain & orientation -> JSON.

    For each selected cell: elevation (median site altitude), mountainess (altitude
    spread), and an 8-way orientation availability vector. Uses dhv_loc.geojson
    (altitude + directions) and takeoff.geojson (orientation flags).
    """
    selected_cells_path = selected_cells_path or str(PROCESSED_DATA_DIR / "selected_cells.json")
    sites_dir = sites_dir or str(FLYBEEPER_SITES_DIR)
    output_path = output_path or str(CELL_TERRAIN_PATH)

    cells: List[str] = json.loads(Path(selected_cells_path).read_text())
    cellset = set(cells)
    sdir = Path(sites_dir)

    # Collect per-cell altitudes and per-site orientation sets.
    alts: Dict[str, List[float]] = {c: [] for c in cells}
    orient_hits: Dict[str, np.ndarray] = {c: np.zeros(8) for c in cells}
    orient_sites: Dict[str, int] = {c: 0 for c in cells}

    dhv = json.loads((sdir / "dhv_loc.geojson").read_text())
    dhv_feats = dhv["features"] if isinstance(dhv, dict) else dhv
    for f in dhv_feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        c = _cell_of(lat, lon)
        if c not in cellset:
            continue
        p = f["properties"]
        if p.get("altitude") is not None:
            alts[c].append(float(p["altitude"]))
        o = _dhv_orientations(p.get("directionsText"))
        if o:
            for k in o:
                orient_hits[c][k] += 1
            orient_sites[c] += 1

    take = json.loads((sdir / "takeoff.geojson").read_text())
    take_feats = take["features"] if isinstance(take, dict) else take
    for f in take_feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        c = _cell_of(lat, lon)
        if c not in cellset:
            continue
        o = _takeoff_orientations(f["properties"])
        if o:
            for k in o:
                orient_hits[c][k] += 1
            orient_sites[c] += 1

    terrain: Dict[str, dict] = {}
    for c in cells:
        a = np.array(alts[c], dtype=float)
        if a.size:
            elevation = float(np.median(a))
            spread = float(np.percentile(a, 90) - np.percentile(a, 10)) if a.size >= 5 else float(a.max() - a.min())
            mountainess = max(0.0, min(1.0, spread / 1000.0))
        else:
            elevation, mountainess = 0.0, 0.0
        n = orient_sites[c]
        orientations = (orient_hits[c] / n).tolist() if n else [0.0] * 8
        terrain[c] = {
            "elevation": elevation,
            "mountainess": mountainess,
            "orientations": orientations,
            "n_alt_sites": int(a.size),
            "n_orient_sites": int(n),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(terrain, indent=2))
    empty = [c for c, v in terrain.items() if v["n_alt_sites"] == 0]
    print(f"✓ Terrain for {len(terrain)} cells -> {output_path}"
          + (f"  ({len(empty)} cells без спотов: {empty})" if empty else ""))
    return terrain


def load_cell_terrain(path: Optional[str] = None) -> Optional[Dict[str, dict]]:
    """Load the per-cell terrain table if it exists, else None."""
    p = Path(path or CELL_TERRAIN_PATH)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def add_terrain_features(rec: dict, terrain_cell: Optional[dict]) -> dict:
    """
    Add static spot-centric terrain (known elevation/mountainess) to a record.

    Shared by the dataset builder and the forecast path so both compute identically.
    Slope-wind features are intentionally NOT added (see slope_wind_alignment: they
    tested as negligible on GFS winds).
    """
    if not terrain_cell:
        return rec
    rec["elevation"] = float(terrain_cell.get("elevation", 0.0))
    rec["mountainess"] = float(terrain_cell.get("mountainess", 0.0))
    return rec
