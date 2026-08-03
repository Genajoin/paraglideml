"""
The cell grid — the one place that knows what a cell id means.

Cells are Δlat 0.75° × Δlon 1°. The latitude step is NOT 1°: a 1°×1° cell is a
rectangle by construction (111 km tall, 111·cos(lat) km wide), so at 46°N it stood
1.45× taller than wide. Mercator is conformal, so that shape is exactly what the map
showed. 0.75 brings the median cell to 1.08 across our 36–55°N belt, and — unlike the
geometric optimum 0.698 — it is a multiple of the GFS 0.25° step, so every cell anchor
still lands exactly on a grid node.

Cell id is ``"{lat0:.2f}_{lon0:d}"`` — the anchor (south-west corner) in degrees, e.g.
``"45.75_11"``, ``"36.00_-6"``. Latitude always carries two decimals *on purpose*: the
previous contract was ``floor(lat)_floor(lon)`` and ``"45_11"`` meant the 45–46° square.
Writing ``"45.00_11"`` guarantees no new id can ever be mistaken for an old one, in either
direction: a stale consumer finds no match and renders nothing instead of confidently
placing a cell half a degree off, and ``cell_anchor`` rejects a legacy id outright rather
than letting ``float("45")`` resolve it to the wrong cell.

Dependency-free (like tiers.py) so the lean `paraglideml[inference]` install and the
training path can both import it.
"""

import math
import re
from typing import List, Tuple

# Grid steps in degrees. LAT_STEP must stay a multiple of the GFS 0.25° spacing.
LAT_STEP = 0.75
LON_STEP = 1.0


def cell_id(lat: float, lon: float) -> str:
    """Cell id containing the point (lat, lon). Idempotent on an anchor."""
    lat0 = math.floor(lat / LAT_STEP) * LAT_STEP
    lon0 = math.floor(lon / LON_STEP) * LON_STEP
    return f"{lat0:.2f}_{int(lon0)}"


# The two decimals are load-bearing, so they are enforced, not merely produced.
# `float("45")` parses happily, so without this a legacy "45_11" would silently
# resolve to the 45.00-45.75 cell — the exact half-degree shift the format exists
# to prevent. Guarding here makes a stale cell list fail loudly on the first read.
_CELL_RE = re.compile(r"^-?\d+\.\d{2}_-?\d+$")


def cell_anchor(cid: str) -> Tuple[float, float]:
    """(lat0, lon0) south-west corner of a cell id. Raises on a malformed or legacy id."""
    if not _CELL_RE.match(cid):
        raise ValueError(
            f"cell id {cid!r} is not in the current format '<lat0 with two decimals>_<lon0>' "
            f"(e.g. '45.75_11'). Ids like '45_11' are the pre-0.75-grid contract and mean "
            f"a different square — regenerate the cell list rather than reusing it."
        )
    lat_s, lon_s = cid.split("_")
    return float(lat_s), float(lon_s)


def cell_center(cid: str) -> Tuple[float, float]:
    lat0, lon0 = cell_anchor(cid)
    return lat0 + LAT_STEP / 2.0, lon0 + LON_STEP / 2.0


def cell_bounds(cid: str) -> Tuple[float, float, float, float]:
    """(lat0, lon0, lat1, lon1) — the cell's half-open [lat0, lat1) x [lon0, lon1)."""
    lat0, lon0 = cell_anchor(cid)
    return lat0, lon0, lat0 + LAT_STEP, lon0 + LON_STEP


def cell_ring(cid: str) -> List[List[float]]:
    """Closed GeoJSON polygon ring [lon, lat] for the cell square."""
    lat0, lon0, lat1, lon1 = cell_bounds(cid)
    return [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]


def cells_bbox(cids: List[str]) -> Tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) covering every cell in the list."""
    bounds = [cell_bounds(c) for c in cids]
    return (
        min(b[1] for b in bounds),
        min(b[0] for b in bounds),
        max(b[3] for b in bounds),
        max(b[2] for b in bounds),
    )


def cells_in_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> List[str]:
    """Every cell whose anchor falls in the bbox — the extraction work-list."""
    out = []
    lat0 = math.floor(lat_min / LAT_STEP) * LAT_STEP
    while lat0 < lat_max:
        lon0 = math.floor(lon_min / LON_STEP) * LON_STEP
        while lon0 < lon_max:
            out.append(f"{lat0:.2f}_{int(lon0)}")
            lon0 += LON_STEP
        lat0 += LAT_STEP
    return out


def contains(cid: str, lat: float, lon: float) -> bool:
    """Half-open membership, matching `cell_id` exactly (no double-counted borders)."""
    lat0, lon0, lat1, lon1 = cell_bounds(cid)
    return lat0 <= lat < lat1 and lon0 <= lon < lon1
