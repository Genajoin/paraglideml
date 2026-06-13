"""
Forecast / inference path: run a trained model on GFS data for a given date.

This is the core of the bot / FlyBeeper inference pipeline. It downloads only the
GRIB messages the model needs (byte-range via the .idx index — ~110 MB instead of
~500 MB per slice), extracts the per-cell features with the same code as the
training builder, and prints per-spot flyability probabilities.

v1 uses the GFS 0.25° analysis (f000) from NOAA S3 — i.e. the real conditions for
the date, the same kind of field the model trained on. This is the right input for
eyeballing "did the model agree with what actually happened" on a recent day.
True multi-day forecasting (lead times f024..f120) is a later addition.
"""

import datetime as dt
import json
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import assets
from .config import EXPERIMENTS_DIR, GFS_CACHE_DIR, GFS_FORECAST_DIR, PROCESSED_DATA_DIR
from .data.dataset_builder import compute_day_features
from .data.gfs_processor import PRESSURE_LEVELS, run_gfs_cache_creation
from .data.terrain import add_terrain_features, load_cell_terrain
from .data.weather_cache import WeatherCache
from .tiers import TIER_LABELS

# torch + the NN model are imported lazily inside run_forecast() so the ordinal /
# library inference path (predict_tiers) stays torch-free — `paraglideml[inference]`.

S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Which GRIB messages the model needs, expressed as (.idx VAR, level) selectors.
_PRESSURE_VARS = {"TMP", "HGT", "RH", "UGRD", "VGRD", "VVEL"}
_PRESSURE_LEVELS_MB = {f"{p} mb" for p in PRESSURE_LEVELS}
_SURFACE_SELECT = {
    ("TMP", "2 m above ground"),
    ("DPT", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("CAPE", "surface"),
    ("CIN", "surface"),
    ("PRES", "surface"),
    ("GUST", "surface"),
    ("TCDC", "entire atmosphere"),
    ("VIS", "surface"),
}


def _want_message(var: str, level: str) -> bool:
    if var in _PRESSURE_VARS and level in _PRESSURE_LEVELS_MB:
        return True
    return (var, level) in _SURFACE_SELECT


def download_gfs_slice(date: dt.date, hour: int, dest: Path, fxx: int = 0) -> bool:
    """
    Download only the needed GRIB messages for one GFS slice via .idx byte ranges.

    Args:
        date: run/valid date (analysis f000 -> valid == run)
        hour: cycle hour (6/12/18)
        dest: output .grb2 path
        fxx: forecast hour (0 = analysis). For forecasts the caller maps a valid
             time to (run, fxx); here we just fetch the requested file.

    Returns True on success, False if the slice isn't available yet.
    """
    ymd = date.strftime("%Y%m%d")
    base = f"{S3_BASE}/gfs.{ymd}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.0p25.f{fxx:03d}"
    try:
        with urllib.request.urlopen(base + ".idx", timeout=30) as r:
            idx_lines = r.read().decode().splitlines()
    except Exception as e:
        print(f"  [{ymd} {hour:02d}z f{fxx:03d}] index unavailable: {e}")
        return False

    entries: List[Tuple[int, str, str]] = []
    for line in idx_lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            start = int(parts[1])
        except ValueError:
            continue
        entries.append((start, parts[3], parts[4]))

    # Build byte ranges for the wanted messages (end = next message's start - 1).
    ranges: List[List[Optional[int]]] = []
    for i, (start, var, level) in enumerate(entries):
        end = entries[i + 1][0] - 1 if i + 1 < len(entries) else None
        if _want_message(var, level):
            ranges.append([start, end])
    if not ranges:
        print(f"  [{ymd} {hour:02d}z] no matching messages in index")
        return False

    # Merge adjacent ranges to cut the number of HTTP requests.
    ranges.sort(key=lambda r: r[0])
    merged: List[List[Optional[int]]] = []
    for s, e in ranges:
        if merged and merged[-1][1] is not None and s <= merged[-1][1] + 1:
            if e is None or e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for s, e in merged:
            rng = f"bytes={s}-{e}" if e is not None else f"bytes={s}-"
            req = urllib.request.Request(base, headers={"Range": rng})
            with urllib.request.urlopen(req, timeout=300) as r:
                chunk = r.read()
                f.write(chunk)
                total += len(chunk)
    print(f"  [{ymd} {hour:02d}z] downloaded {total / 1e6:.1f} MB ({len(merged)} ranges)")
    return True


def _resolve_experiment(experiment: Optional[str]) -> Path:
    """Return the experiment dir (explicit name, or the latest NN experiment)."""
    if experiment:
        exp_dir = EXPERIMENTS_DIR / experiment
        if not (exp_dir / "model.pth").exists():
            raise FileNotFoundError(f"No NN model.pth in {exp_dir}")
        return exp_dir
    candidates = [
        d for d in sorted(EXPERIMENTS_DIR.iterdir()) if (d / "model.pth").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No NN experiments with model.pth in {EXPERIMENTS_DIR}")
    return candidates[-1]


def _fetch_and_extract(date_str: str, grib_root: Path) -> None:
    """Download the 06/12/18 GFS slices for a date and extract per-cell NPZ cache.

    Prefers the analysis (real conditions); if not posted yet (today/future), falls
    back to the forecast valid at that time from the same day's 00z run. Shared by
    the binary and ordinal forecast paths.
    """
    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    print(f">>> Fetching GFS for {date_str} ...")
    sources = {}
    for hour in (6, 12, 18):
        dest = (
            grib_root
            / target.strftime("%Y-%m")
            / f"gfsanl_3_{target.strftime('%Y%m%d')}_{hour:02d}00_000.grb2"
        )
        if dest.exists() and dest.stat().st_size > 0:
            sources[hour] = "cached"
        elif download_gfs_slice(target, hour, dest, fxx=0):
            sources[hour] = "analysis"
        elif download_gfs_slice(target, 0, dest, fxx=hour):
            sources[hour] = f"forecast(00z+{hour}h)"
    print(f"  slices: {sources}")

    print(">>> Extracting features to cache ...")
    run_gfs_cache_creation(
        dates=f"{date_str}:{date_str}",
        bbox="6.0,43.0,17.0,49.0",
        source_dir=grib_root,
        output_dir=GFS_CACHE_DIR,
        force=True,
    )


def _feature_vector(rec: dict, feature_names: List[str], terrain_cell: Optional[dict]) -> np.ndarray:
    """Build the model input row, injecting spot-centric terrain features.

    compute_day_features returns weather only; elevation/mountainess/slope-wind are
    supplied from cell_terrain.json at inference via the same add_slope_features the
    dataset builder uses at training time (otherwise they'd read as 0.0).
    """
    rec = add_terrain_features(dict(rec), terrain_cell)
    return np.array([[float(rec.get(f, 0.0)) for f in feature_names]], dtype=np.float32)


def _resolve_ordinal_experiment(experiment: Optional[str]) -> Path:
    """Return the ordinal experiment dir (explicit, or latest with model_good.joblib)."""
    if experiment:
        exp_dir = EXPERIMENTS_DIR / experiment
        if not (exp_dir / "model_good.joblib").exists():
            raise FileNotFoundError(f"No ordinal model_good.joblib in {exp_dir}")
        return exp_dir
    candidates = [d for d in sorted(EXPERIMENTS_DIR.iterdir()) if (d / "model_good.joblib").exists()]
    if not candidates:
        raise FileNotFoundError(f"No ordinal experiments (model_good.joblib) in {EXPERIMENTS_DIR}")
    return candidates[-1]


def run_forecast(
    date_str: str,
    experiment: Optional[str] = None,
    selected_cells_path: Optional[Path] = None,
    grib_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download GFS for a date, run the model, return a per-cell prediction table."""
    import torch  # lazy: only the NN path needs torch (keeps ordinal inference lean)

    from .multiregional import MultiRegionalModel

    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    grib_root = Path(grib_dir) if grib_dir else GFS_FORECAST_DIR

    _fetch_and_extract(date_str, grib_root)

    # 3. Load model artifacts.
    exp_dir = _resolve_experiment(experiment)
    cfg = json.loads((exp_dir / "config.json").read_text())
    feature_names = (exp_dir / "features.txt").read_text().splitlines()
    region_map = json.loads((exp_dir / "region_mapping.json").read_text())
    scaler = joblib.load(exp_dir / "scaler.pkl")
    threshold = float(cfg.get("threshold", 0.5))

    model = MultiRegionalModel(
        input_dim=len(feature_names),
        num_regions=cfg["num_regions"],
        embedding_dim=cfg["regional_embedding_dim"],
        dropout_rate=cfg["dropout_rate"],
    )
    model.load_state_dict(torch.load(exp_dir / "model.pth", map_location="cpu"))
    model.eval()
    print(f">>> Model: {exp_dir.name} ({len(feature_names)} features, threshold={threshold:.2f})")

    # 4. Predict per cell.
    if selected_cells_path is None:
        selected_cells_path = PROCESSED_DATA_DIR / "selected_cells.json"
    cells = json.loads(Path(selected_cells_path).read_text())
    cache = WeatherCache(cache_root=str(GFS_CACHE_DIR))
    terrain = load_cell_terrain() or {}
    ts = pd.Timestamp(target)

    rows = []
    for cell in cells:
        lat, lon = map(int, cell.split("_"))
        rec = compute_day_features(cache, lat, lon, ts)
        if rec is None:
            continue
        x = _feature_vector(rec, feature_names, terrain.get(cell))
        xs = scaler.transform(x)
        region = int(region_map.get(cell, 0))
        with torch.no_grad():
            logit = model(
                torch.tensor(xs, dtype=torch.float32),
                torch.tensor([region], dtype=torch.long),
            )
            prob = float(torch.sigmoid(logit).item())
        rows.append(
            {
                "cell": cell,
                "lat": lat,
                "lon": lon,
                "region": region,
                "probability": prob,
                "flyable": int(prob > threshold),
            }
        )

    if not rows:
        print(f"\n=== Flyability forecast for {date_str} ===")
        print("  No cells with data — the 12 UTC slice is unavailable for this date.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)

    # 5. Print a readable table.
    print(f"\n=== Flyability forecast for {date_str}  (threshold {threshold:.2f}) ===")
    if df.empty:
        print("  No cells with data (12 UTC slice missing?).")
    else:
        for _, r in df.iterrows():
            bar = "#" * int(round(r["probability"] * 20))
            mark = "✈" if r["flyable"] else " "
            print(f"  {mark} {r['cell']:>6}  r{r['region']}  {r['probability']*100:5.1f}%  {bar}")
        n_fly = int(df["flyable"].sum())
        print(f"\n  {n_fly}/{len(df)} cells above threshold "
              f"(regional signal: {'STRONG' if n_fly >= 5 else 'weak — treat with caution'}).")
    return df


def _load_ordinal_models(model_dir: Path):
    """Load the ordinal trio + calibrators + feature order from a model dir."""
    feature_names = [l for l in (model_dir / "features.txt").read_text().splitlines() if l.strip()]
    models = {lbl: joblib.load(model_dir / f"model_{lbl}.joblib") for lbl in TIER_LABELS}
    calibrators = {lbl: joblib.load(model_dir / f"calibrator_{lbl}.joblib") for lbl in TIER_LABELS}
    return feature_names, models, calibrators


def predict_tiers(
    date_str: str,
    model_dir: Optional[Path] = None,
    cells: Optional[List[str]] = None,
    grib_dir: Optional[str] = None,
) -> List[dict]:
    """Per-cell cumulative tier probabilities for one date — the library inference API.

    Returns one dict per cell: {cell, lat, lon, date, p_flyable, p_good, p_epic}, with
    probabilities calibrated and clamped non-increasing (P>=flyable >= P>=good >= P>=epic).
    No printing — the bot / pipeline / GeoJSON builder consume the returned list.

    Model defaults to the bundled exp_056 (override via PARAGLIDEML_MODEL_DIR or model_dir).
    Downloads + extracts the GFS slice for the date using the shared fetch path.
    """
    mdir = Path(model_dir) if model_dir else assets.model_dir()
    feature_names, models, calibrators = _load_ordinal_models(mdir)

    grib_root = Path(grib_dir) if grib_dir else GFS_FORECAST_DIR
    _fetch_and_extract(date_str, grib_root)
    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    if cells is None:
        cells = json.loads(assets.cells_path().read_text())
    terrain = load_cell_terrain(str(assets.terrain_path())) or {}
    cache = WeatherCache(cache_root=str(GFS_CACHE_DIR))
    ts = pd.Timestamp(target)

    rows: List[dict] = []
    for cell in cells:
        lat, lon = map(int, cell.split("_"))
        rec = compute_day_features(cache, lat, lon, ts)
        if rec is None:
            continue
        x = _feature_vector(rec, feature_names, terrain.get(cell))
        probs = [float(calibrators[lbl].transform(models[lbl].predict_proba(x)[:, 1])[0]) for lbl in TIER_LABELS]
        for i in range(1, len(probs)):  # enforce monotonicity across tiers
            probs[i] = min(probs[i], probs[i - 1])
        row = {"cell": cell, "lat": lat, "lon": lon, "date": date_str}
        row.update({f"p_{lbl}": p for lbl, p in zip(TIER_LABELS, probs)})
        rows.append(row)
    return rows


def tiers_to_geojson(rows: List[dict], date_str: Optional[str] = None) -> dict:
    """Render tier rows as a GeoJSON FeatureCollection of honest 1-degree squares.

    Each GFS cell `lat_lon` covers [lon, lon+1] x [lat, lat+1] degrees; we emit that exact
    square polygon (not a point) so the map never implies sub-cell precision. This is the
    artifact contract: the pipeline writes it to R2, the Worker serves it, the map layer
    and the GitHub snapshot demo both render it.
    """
    features = []
    for r in rows:
        lat, lon = int(r["lat"]), int(r["lon"])
        ring = [[lon, lat], [lon + 1, lat], [lon + 1, lat + 1], [lon, lat + 1], [lon, lat]]
        props = {
            "cell": r["cell"],
            "p_flyable": round(float(r.get("p_flyable", 0.0)), 4),
            "p_good": round(float(r.get("p_good", 0.0)), 4),
            "p_epic": round(float(r.get("p_epic", 0.0)), 4),
        }
        d = date_str or r.get("date")
        if d:
            props["date"] = d
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def run_ordinal_forecast(
    date_str: str,
    experiment: Optional[str] = None,
    selected_cells_path: Optional[Path] = None,
    grib_dir: Optional[str] = None,
    push_threshold: float = 0.5,
    model_dir: Optional[str] = None,
    geojson_out: Optional[str] = None,
) -> pd.DataFrame:
    """CLI-facing wrapper: compute tiers, print the table, optionally write GeoJSON.

    Model source precedence: --experiment (dev, scans experiments/) > model_dir > bundled
    exp_056 (default). Cumulative calibrated probabilities; the bot pushes on P(>=good).
    Uses the GFS analysis for a past date — ideal for eyeballing the tiers vs what flew.
    """
    mdir: Optional[Path] = None
    if experiment:
        mdir = _resolve_ordinal_experiment(experiment)
    elif model_dir:
        mdir = Path(model_dir)
    print(f">>> Ordinal model: {mdir if mdir else assets.model_dir()} (tiers {TIER_LABELS})")

    cells = json.loads(Path(selected_cells_path).read_text()) if selected_cells_path else None
    rows = predict_tiers(date_str, model_dir=mdir, cells=cells, grib_dir=grib_dir)

    if not rows:
        print(f"\n=== Tier forecast for {date_str} ===")
        print("  No cells with data — the 12 UTC slice is unavailable for this date.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("p_good", ascending=False).reset_index(drop=True)

    print(f"\n=== Flight-quality tiers for {date_str}  (P>=flyable / good / epic) ===")
    for _, r in df.iterrows():
        bar = "#" * int(round(r["p_good"] * 20))
        push = "→PUSH" if r["p_good"] >= push_threshold else "     "
        print(f"  {r['cell']:>6}  "
              f"fly {r['p_flyable']*100:4.0f}%  good {r['p_good']*100:4.0f}%  "
              f"epic {r['p_epic']*100:4.0f}%  {push} {bar}")
    n_good = int((df["p_good"] >= push_threshold).sum())
    n_epic = int((df["p_epic"] >= push_threshold).sum())
    print(f"\n  {n_good}/{len(df)} cells P(>=good)>={push_threshold:.0%}, "
          f"{n_epic} P(>=epic)>={push_threshold:.0%} "
          f"(regional signal: {'STRONG' if n_good >= 5 else 'weak — treat with caution'}).")

    if geojson_out:
        gj = tiers_to_geojson(rows, date_str)
        Path(geojson_out).write_text(json.dumps(gj))
        print(f"\n>>> Wrote GeoJSON ({len(gj['features'])} cells) to {geojson_out}")
    return df
