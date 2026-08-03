"""
Render the per-cell tier forecast as a map PNG.

The published artifact is GeoJSON meant for the MapLibre layer; this is the same
content as a static picture — for a chat/report/README, or for eyeballing a run
without opening the web map.

Plotting deps (matplotlib + cartopy) live in the `train` extra, not in the lean
inference install, so they are imported lazily: `pip install paraglideml[train]`.

    paraglideml forecast-map --date 2026-08-03 --out map.png
    paraglideml forecast-map --artifact out/flyability/20260803/forecast.json --out map.png
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

TIERS = {
    "flyable": ("p_flyable", "≥ 15 км", "Вероятность лётного дня"),
    "good": ("p_good", "≥ 50 км", "Вероятность хорошего XC-дня"),
    "epic": ("p_epic", "≥ 100 км", "Вероятность выдающегося XC-дня"),
}

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#1a1a18", "#4a4a46", "#84847e"
# Sequential blue ramp (palette steps 100/200/350/500/700): light -> dark = low -> high.
# Sequential encoding wants one hue and monotone lightness; never a rainbow.
BINS = [0.0, 0.10, 0.25, 0.45, 0.65, 1.01]
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
BIN_LABELS = ["< 10 %", "10–25 %", "25–45 %", "45–65 %", "> 65 %"]


def _bin_colour(p: float) -> str:
    for i in range(len(BINS) - 1):
        if BINS[i] <= p < BINS[i + 1]:
            return RAMP[i]
    return RAMP[-1]


def rows_from_artifact(path: str) -> List[dict]:
    """Read cell rows back out of a published flybeeper.flyability.v1 artifact."""
    doc = json.loads(Path(path).read_text())
    gj = doc.get("geojson", doc)
    rows = []
    for f in gj["features"]:
        pr = f["properties"]
        lon, lat = f["geometry"]["coordinates"][0][0]
        rows.append({**pr, "lat": int(lat), "lon": int(lon)})
    return rows


def busiest_launches(
    data_dir: str = "data/flights", years: Optional[List[int]] = None, bbox: Optional[str] = None
) -> Dict[str, str]:
    """Map cell_id -> busiest launch site name, for human-readable labels.

    Purely cosmetic: a reader knows "Bassano", not "45_11". Returns {} if the world
    flight export isn't present, and the caller falls back to cell ids.
    """
    try:
        import numpy as np

        from .data.flight_parsing import load_world_flights

        df = load_world_flights(data_dir=data_dir, bbox=bbox, years=years or [2024, 2025])
    except Exception:
        return {}
    df = df[df["points"] >= 10]
    cell = (
        np.floor(df["takeoff_lat"]).astype(int).astype(str)
        + "_"
        + np.floor(df["takeoff_lon"]).astype(int).astype(str)
    )
    counts = df.assign(cell=cell).groupby(["cell", "takeoff_name"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False).drop_duplicates("cell")
    return dict(zip(counts["cell"], counts["takeoff_name"]))


def render_tier_map(
    rows: List[dict],
    date_str: str,
    out_path: str,
    tier: str = "good",
    labels: Optional[Dict[str, str]] = None,
    top_n: int = 10,
    subtitle: Optional[str] = None,
    footer: Optional[str] = None,
) -> str:
    """Draw the cells as 1-degree squares coloured by the chosen tier probability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "cartopy is required for the map: pip install 'paraglideml[train]'"
        ) from e

    key, tier_km, title_stem = TIERS[tier]
    labels = labels or {}
    rows = [r for r in rows if r.get("date", date_str) == date_str]
    if not rows:
        raise SystemExit(f"no rows for {date_str}")
    ranked = sorted(rows, key=lambda r: -r[key])[:top_n]

    lons = [r["lon"] for r in rows]
    lats = [r["lat"] for r in rows]
    pad_x, pad_y = 2.5, 1.5
    extent = [min(lons) - pad_x, max(lons) + 1 + pad_x, min(lats) - pad_y, max(lats) + 1 + pad_y]
    span_x, span_y = extent[1] - extent[0], extent[3] - extent[2]

    map_w = 10.6
    fig_h = max(5.0, map_w * span_y / span_x / 0.845 + 1.1)
    fig = plt.figure(figsize=(map_w / 0.70, fig_h), facecolor=SURFACE)
    ax = fig.add_axes((0.015, 0.045, 0.70, 0.845), projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e9eef4", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f2ef", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#b6b6b0", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#cbcbc5", zorder=4)
    for s in ax.spines.values():
        s.set_visible(False)

    # Honest 1-degree squares — the artifact contract; never implying point accuracy.
    # The surface-coloured inset keeps neighbouring fills from merging into one blob.
    for r in rows:
        ax.add_patch(
            Rectangle(
                (r["lon"] + 0.05, r["lat"] + 0.05), 0.90, 0.90,
                facecolor=_bin_colour(r[key]), edgecolor=SURFACE, lw=1.3,
                transform=ccrs.PlateCarree(), zorder=3,
            )
        )

    # Numbered badges keyed to the table, not inline text: in the Alps the cells sit
    # shoulder to shoulder and any inline label collides with its neighbours. The badge
    # is small and corner-anchored so it doesn't wash out the colour being ranked.
    for i, r in enumerate(ranked, 1):
        bx, by = r["lon"] + 0.26, r["lat"] + 0.74
        ax.add_patch(
            Circle((bx, by), 0.23, facecolor=SURFACE, edgecolor="#5a5a55", lw=0.8,
                   transform=ccrs.PlateCarree(), zorder=6)
        )
        ax.text(bx, by, str(i), fontsize=6.6, color=INK, ha="center", va="center",
                weight="bold", zorder=7, transform=ccrs.PlateCarree())

    fig.text(0.015, 1 - 0.52 / fig_h, f"{title_stem} ({tier_km}) — {date_str}",
             fontsize=18, color=INK, weight="semibold", va="top")
    fig.text(0.015, 1 - 0.90 / fig_h, subtitle or f"{len(rows)} ячеек",
             fontsize=10.5, color=MUTED, va="top")

    # Legend: identity is never colour-alone, so the ramp is spelled out in numbers.
    lg = fig.add_axes((0.735, 0.60, 0.10, 0.29))
    lg.axis("off")
    lg.text(0, 1.02, f"P({tier_km})", fontsize=10.5, color=INK, weight="semibold", va="top")
    for i, (c, lab) in enumerate(zip(RAMP, BIN_LABELS)):
        y = 0.80 - i * 0.155
        lg.add_patch(Rectangle((0, y - 0.075), 0.30, 0.10, facecolor=c,
                               edgecolor="#e2e2dc", lw=0.8, transform=lg.transAxes))
        lg.text(0.40, y - 0.025, lab, fontsize=9.5, color=INK2, va="center")

    # Ranked table — the badge key, and the relief the pale steps' low contrast obliges.
    tb = fig.add_axes((0.735, 0.045, 0.255, 0.50))
    tb.axis("off")
    tb.text(0, 1.06, f"Лучшие {len(ranked)} ячеек", fontsize=10.5, color=INK,
            weight="semibold", va="top")
    tb.text(0, 0.965, "   старт           ≥15  ≥50 ≥100 км", fontsize=8, color=MUTED,
            va="top", family="monospace")
    for i, r in enumerate(ranked):
        name = labels.get(r["cell"], r["cell"])[:14]
        tb.text(0, 0.885 - i * (0.88 / max(top_n, 1)),
                f"{i+1:>2d} {name:<14s} {r['p_flyable']:>3.0%} "
                f"{r['p_good']:>4.0%} {r['p_epic']:>4.0%}",
                fontsize=8.6, color=INK2, va="top", family="monospace")

    fig.text(0.015, 0.012, footer or "Ячейка 1° ≈ 110 км — квадрат честный, не точка.",
             fontsize=8.5, color=MUTED)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out_path
