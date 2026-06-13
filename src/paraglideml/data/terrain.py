"""
Spot-centric terrain features (launch-point elevation & mountainess).

The weather features are extracted at the 1° cell, but a cell's *flying* happens at
the launch sites, which can sit on a mountain edge while the cell as a whole is mostly
plain (e.g. cell 45_11: centre ~80 m / mountainess 0.31, but the launch centroid is at
~680 m / mountainess 1.0). Feeding cell-centre terrain would mislabel such cells as
flat. So we anchor terrain at the **launch centroid** — the median takeoff coordinate
of the cell's quality flights — and read elevation + mountainess there.

This is a one-time extraction (`paraglideml data terrain`) that reads the heavy
elevation raster once and writes a small per-cell JSON. The dataset builder and the
forecast path consume only that JSON, so neither needs rasterio at runtime.

Mountainess follows the archive definition: sample a grid in a radius around the
point, mountainess = clamp((max_elev - min_elev) / 800, 0, 1).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import CELL_TERRAIN_PATH, ELEVATION_TIF, FLIGHTS_DIR, PROCESSED_DATA_DIR
from .flight_parsing import load_flights_to_dataframe


class GeoTiffElevation:
    """Minimal GeoTIFF elevation sampler (elevation + mountainess)."""

    def __init__(self, tif_path: Path):
        import rasterio  # local import: only the terrain extraction needs rasterio

        self._rio = rasterio
        self.ds = rasterio.open(tif_path)
        self.nodata = self.ds.nodata

    def elevation(self, lat: float, lon: float) -> Optional[float]:
        from rasterio.transform import rowcol
        from rasterio.windows import Window

        row, col = rowcol(self.ds.transform, lon, lat)
        if not (0 <= row < self.ds.height and 0 <= col < self.ds.width):
            return None
        val = self.ds.read(1, window=Window(col, row, 1, 1))
        if val.size == 0:
            return None
        e = float(val[0, 0])
        if self.nodata is not None and e == self.nodata:
            return None
        return e

    def mountainess(self, lat: float, lon: float, grid_size: int = 5, radius_km: float = 5.0) -> float:
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
        step_lat = (radius_km * 2) / (grid_size - 1) / km_per_deg_lat
        step_lon = (radius_km * 2) / (grid_size - 1) / km_per_deg_lon
        start_lat = lat - radius_km / km_per_deg_lat
        start_lon = lon - radius_km / km_per_deg_lon
        elevs: List[float] = []
        for i in range(grid_size):
            for j in range(grid_size):
                e = self.elevation(start_lat + i * step_lat, start_lon + j * step_lon)
                if e is not None:
                    elevs.append(e)
        if not elevs:
            return 0.0
        return max(0.0, min(1.0, (max(elevs) - min(elevs)) / 800.0))

    def close(self):
        self.ds.close()


def build_cell_terrain(
    selected_cells_path: Optional[str] = None,
    flights_dir: Optional[str] = None,
    elevation_tif: Optional[str] = None,
    output_path: Optional[str] = None,
    min_xc_points: int = 10,
) -> Dict[str, dict]:
    """
    Compute launch-centroid elevation & mountainess per selected cell -> JSON.

    For each cell, the launch centroid is the median takeoff coordinate over the
    cell's quality flights; terrain is sampled there (falling back to the cell
    centre if a cell has no geolocated flights).
    """
    selected_cells_path = selected_cells_path or str(PROCESSED_DATA_DIR / "selected_cells.json")
    flights_dir = flights_dir or str(FLIGHTS_DIR)
    elevation_tif = elevation_tif or str(ELEVATION_TIF)
    output_path = output_path or str(CELL_TERRAIN_PATH)

    cells: List[str] = json.loads(Path(selected_cells_path).read_text())
    print(f"Computing terrain for {len(cells)} cells from {Path(elevation_tif).name} ...")

    df = load_flights_to_dataframe(data_dir=flights_dir).dropna(subset=["takeoff_lat", "takeoff_lon"])
    if min_xc_points > 0:
        df = df[df["points"] >= min_xc_points]

    reader = GeoTiffElevation(Path(elevation_tif))
    terrain: Dict[str, dict] = {}
    try:
        for cell in cells:
            lat, lon = map(int, cell.split("_"))
            cc_lat, cc_lon = lat + 0.5, lon + 0.5
            sub = df[
                df["takeoff_lat"].between(cc_lat - 0.5, cc_lat + 0.5)
                & df["takeoff_lon"].between(cc_lon - 0.5, cc_lon + 0.5)
            ]
            if len(sub):
                launch_lat = float(sub["takeoff_lat"].median())
                launch_lon = float(sub["takeoff_lon"].median())
                source = "launch_centroid"
            else:
                launch_lat, launch_lon, source = cc_lat, cc_lon, "cell_center_fallback"

            elev = reader.elevation(launch_lat, launch_lon)
            terrain[cell] = {
                "launch_lat": launch_lat,
                "launch_lon": launch_lon,
                "elevation": float(elev) if elev is not None else 0.0,
                "mountainess": float(reader.mountainess(launch_lat, launch_lon)),
                "n_flights": int(len(sub)),
                "source": source,
            }
    finally:
        reader.close()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(terrain, indent=2))
    n_fallback = sum(1 for v in terrain.values() if v["source"] != "launch_centroid")
    print(f"✓ Terrain for {len(terrain)} cells -> {output_path} ({n_fallback} cell-center fallbacks)")
    return terrain


def load_cell_terrain(path: Optional[str] = None) -> Optional[Dict[str, dict]]:
    """Load the per-cell terrain table if it exists, else None."""
    p = Path(path or CELL_TERRAIN_PATH)
    if not p.exists():
        return None
    return json.loads(p.read_text())
