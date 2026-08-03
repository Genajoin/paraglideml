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
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
from .grid import LAT_STEP, LON_STEP, cell_anchor, cell_ring, cells_bbox
from .tiers import TIER_LABELS

# torch + the NN model are imported lazily inside run_forecast() so the ordinal /
# library inference path (predict_tiers) stays torch-free — `paraglideml[inference]`.

S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Concurrent byte-range connections per slice. 8 measured ~45 MB/s against S3;
# 16 was slower (throttling), 1 gives ~1.6 MB/s. Override for a fatter pipe.
RANGE_FETCH_WORKERS = int(os.getenv("PARAGLIDEML_RANGE_WORKERS", "8"))

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

    # Fetch the ranges concurrently, then write them back in offset order. The ~66
    # ranges of a slice are latency-bound, not bandwidth-bound: serially they run at
    # ~1.6 MB/s (~65 s per 100 MB slice), with 8 connections at ~45 MB/s. Order matters
    # — pygrib reads the partial file as a GRIB stream, so messages must stay ascending.
    def _fetch(rng_pair):
        s, e = rng_pair
        rng = f"bytes={s}-{e}" if e is not None else f"bytes={s}-"
        req = urllib.request.Request(base, headers={"Range": rng})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=RANGE_FETCH_WORKERS) as ex:
        chunks = list(ex.map(_fetch, merged))
    total = sum(len(c) for c in chunks)
    with open(dest, "wb") as f:
        for chunk in chunks:
            f.write(chunk)
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


def _coerce_date(x) -> Optional[dt.date]:
    """Accept None / 'YYYY-MM-DD' / datetime.date and return a date (or None)."""
    if x is None:
        return None
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    return dt.datetime.strptime(str(x), "%Y-%m-%d").date()


def _fetch_and_extract(
    date_str: str,
    grib_root: Path,
    run_date: Optional[dt.date] = None,
    cache_root: Optional[Path] = None,
    run_cycle: int = 0,
    cells: Optional[List[str]] = None,
) -> Path:
    """Download the 06/12/18 GFS slices for a date, extract per-cell NPZ, return cache root.

    Analysis mode (run_date None or >= target with run_cycle 0): real conditions f000, with a
    same-day 00z forecast fallback if the analysis isn't posted yet — ideal for eyeballing a
    past/today date. Forecast mode (run_date < target, or run_date == target with a fresh
    run_cycle > 0): the FORECAST valid at the target from the `run_date` `run_cycle`z run
    (fxx = (target-run_date).days*24 + valid_hour - run_cycle) — what the bot/pipeline has from
    the latest issued GFS cycle. A per-hour fallback to analysis / 00z covers a valid hour that
    has already passed relative to the cycle (fxx < 0, only possible for today).

    `run_cycle` is the GFS cycle hour (0/6/12/18); the default 0 reproduces the prior 00z-only
    behaviour exactly. Forecast GRIB and cache are keyed by run date *and cycle* (`run{YYYYMMDD}{HH}`)
    so a 06z run never clobbers the same day's 00z cache, and the training analysis cache stays
    clean. Returns the cache root the caller should read from.
    """
    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    grib_root = Path(grib_root)
    cache_root = Path(cache_root) if cache_root else GFS_CACHE_DIR
    forecast = run_date is not None and (run_date < target or (run_date == target and run_cycle > 0))
    lead = (target - run_date).days if forecast else 0

    if forecast:
        grib_root = grib_root / f"run{run_date:%Y%m%d}{run_cycle:02d}"
        cache_root = GFS_CACHE_DIR / "forecast" / f"run{run_date:%Y%m%d}{run_cycle:02d}"
        print(f">>> Fetching GFS FORECAST for {date_str} (run {run_date:%Y-%m-%d} {run_cycle:02d}z, +{lead}d) ...")
    else:
        print(f">>> Fetching GFS for {date_str} (analysis) ...")

    sources = {}
    for hour in (6, 12, 18):
        dest = grib_root / target.strftime("%Y-%m") / f"gfsanl_3_{target:%Y%m%d}_{hour:02d}00_000.grb2"
        if dest.exists() and dest.stat().st_size > 0:
            sources[hour] = "cached"
        elif forecast:
            fxx = lead * 24 + hour - run_cycle  # lead == (target - run_date).days here
            if fxx >= 0 and download_gfs_slice(run_date, run_cycle, dest, fxx=fxx):
                sources[hour] = f"fcst({run_cycle:02d}z+{fxx}h)"
            elif download_gfs_slice(target, hour, dest, fxx=0):
                sources[hour] = "analysis"
            elif download_gfs_slice(target, 0, dest, fxx=hour):
                sources[hour] = f"forecast(00z+{hour}h)"
        elif download_gfs_slice(target, hour, dest, fxx=0):
            sources[hour] = "analysis"
        elif download_gfs_slice(target, 0, dest, fxx=hour):
            sources[hour] = f"forecast(00z+{hour}h)"
    print(f"  slices: {sources}")

    # Extraction bbox follows the cells actually being scored. It used to be the
    # hardcoded Alpine box, which silently dropped every cell outside it — with the
    # 132-cell European set that was 85 cells returning no features at all.
    if cells:
        lon_min, lat_min, lon_max, lat_max = cells_bbox(cells)
        bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    else:
        bbox = "6.0,43.0,17.0,49.0"

    print(">>> Extracting features to cache ...")
    run_gfs_cache_creation(
        dates=f"{date_str}:{date_str}",
        bbox=bbox,
        source_dir=grib_root,
        output_dir=cache_root,
        force=True,
    )
    return cache_root


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
        lat, lon = cell_anchor(cell)
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
    run_date=None,
    run_cycle: int = 0,
) -> List[dict]:
    """Per-cell cumulative tier probabilities for one date — the library inference API.

    Returns one dict per cell: {cell, lat, lon, date, lead, p_flyable, p_good, p_epic}, with
    probabilities calibrated and clamped non-increasing (P>=flyable >= P>=good >= P>=epic).
    No printing — the bot / pipeline / GeoJSON builder consume the returned list.

    Model defaults to the bundled exp_056 (override via PARAGLIDEML_MODEL_DIR or model_dir).
    run_date: None/'YYYY-MM-DD'/date. If a run_date earlier than the target is given, the
    GFS *forecast* valid at the date from that run is used (lead = days ahead) — the bot's
    reality; otherwise the analysis (real conditions, for eyeballing a past day).
    run_cycle: GFS cycle hour (0/6/12/18). Default 0 = the 00z cycle (prior behaviour, 1:1).
    Pass 6/12/18 to score today's forecast from the freshest issued cycle (FlyBeeper refreshes
    every 6h); with run_date == today and run_cycle > 0 the target day is fetched as a forecast.
    """
    mdir = Path(model_dir) if model_dir else assets.model_dir()
    feature_names, models, calibrators = _load_ordinal_models(mdir)

    grib_root = Path(grib_dir) if grib_dir else GFS_FORECAST_DIR
    rd = _coerce_date(run_date)
    if cells is None:
        cells = json.loads(assets.cells_path().read_text())
    cache_root = _fetch_and_extract(
        date_str, grib_root, run_date=rd, run_cycle=run_cycle, cells=cells
    )
    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    lead = (target - rd).days if (rd and rd < target) else 0

    terrain = load_cell_terrain(str(assets.terrain_path())) or {}
    cache = WeatherCache(cache_root=str(cache_root))
    ts = pd.Timestamp(target)

    rows: List[dict] = []
    for cell in cells:
        lat, lon = cell_anchor(cell)
        rec = compute_day_features(cache, lat, lon, ts)
        if rec is None:
            continue
        x = _feature_vector(rec, feature_names, terrain.get(cell))
        probs = [float(calibrators[lbl].transform(models[lbl].predict_proba(x)[:, 1])[0]) for lbl in TIER_LABELS]
        for i in range(1, len(probs)):  # enforce monotonicity across tiers
            probs[i] = min(probs[i], probs[i - 1])
        row = {"cell": cell, "lat": lat, "lon": lon, "date": date_str, "lead": lead}
        row.update({f"p_{lbl}": p for lbl, p in zip(TIER_LABELS, probs)})
        rows.append(row)
    return rows


def forecast_window(
    run_date,
    days: int = 3,
    model_dir: Optional[Path] = None,
    cells: Optional[List[str]] = None,
    grib_dir: Optional[str] = None,
    run_cycle: int = 0,
    start_lead: int = 1,
) -> List[dict]:
    """Multi-day artifact: tier probabilities per cell for run_date+start_lead .. run_date+days.

    Each target day is scored from its own forecast lead-time off the run_date `run_cycle`z
    cycle (the production reality — one issue, N days ahead). Returns the concatenated per-cell
    rows (each tagged with date + lead); feed to tiers_to_geojson for the map/R2 artifact.

    run_cycle: GFS cycle hour (0/6/12/18); default 0 = 00z (prior behaviour). start_lead: first
    lead day; default 1 (tomorrow onward). Pass start_lead=0 to also include today, scored from
    the fresh `run_cycle`z cycle — the 6-hourly refresh path. Defaults (0, 1) reproduce 1:1.
    """
    run = _coerce_date(run_date)
    rows: List[dict] = []
    for lead in range(start_lead, days + 1):
        target = (run + dt.timedelta(days=lead)).strftime("%Y-%m-%d")
        rows.extend(predict_tiers(target, model_dir=model_dir, cells=cells, grib_dir=grib_dir, run_date=run, run_cycle=run_cycle))
    return rows


def tiers_to_geojson(rows: List[dict], date_str: Optional[str] = None) -> dict:
    """Render tier rows as a GeoJSON FeatureCollection of honest cell rectangles.

    Each cell `lat0_lon0` covers [lon0, lon0+LON_STEP] x [lat0, lat0+LAT_STEP] degrees; we
    emit that exact polygon (not a point) so the map never implies sub-cell precision. This
    is the artifact contract: the pipeline writes it to R2, the Worker serves it, the map
    layer and the GitHub snapshot demo both render it. Consumers must read the cell size
    from `cell_lat_degrees` / `cell_lon_degrees` on the wrapper, never assume 1°.
    """
    features = []
    for r in rows:
        ring = cell_ring(r["cell"])
        props = {
            "cell": r["cell"],
            "p_flyable": round(float(r.get("p_flyable", 0.0)), 4),
            "p_good": round(float(r.get("p_good", 0.0)), 4),
            "p_epic": round(float(r.get("p_epic", 0.0)), 4),
        }
        d = date_str or r.get("date")
        if d:
            props["date"] = d
        if r.get("lead") is not None:
            props["lead"] = int(r["lead"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    # The grid travels WITH the data. Cell size stopped being 1° and a consumer that
    # infers geometry from the id (the storm overlay used to) needs to read it, not guess.
    return {
        "type": "FeatureCollection",
        "cell_lat_degrees": LAT_STEP,
        "cell_lon_degrees": LON_STEP,
        "features": features,
    }


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
