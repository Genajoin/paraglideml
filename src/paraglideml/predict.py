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
import torch

from .config import EXPERIMENTS_DIR, GFS_CACHE_DIR, PROCESSED_DATA_DIR
from .data.dataset_builder import compute_day_features
from .data.gfs_processor import PRESSURE_LEVELS, run_gfs_cache_creation
from .data.weather_cache import WeatherCache
from .multiregional import MultiRegionalModel

S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Local landing dir for freshly downloaded GRIB. NOT GFS_ANL_DIR — that is a
# symlink to an external backup that may be unmounted; forecast downloads must be
# writable locally and independent of the training archive.
FORECAST_GRIB_DIR = GFS_CACHE_DIR.parent / "forecast_grib"

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


def run_forecast(
    date_str: str,
    experiment: Optional[str] = None,
    selected_cells_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download GFS for a date, run the model, return a per-cell prediction table."""
    target = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    # 1. Fetch the 06/12/18 slices (byte-range). Prefer the analysis (real
    #    conditions); if it isn't posted yet (today/future), fall back to the
    #    forecast valid at that time from the same day's 00z run (f006/f012/f018).
    print(f">>> Fetching GFS for {date_str} ...")
    sources = {}
    for hour in (6, 12, 18):
        dest = (
            FORECAST_GRIB_DIR
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

    # 2. Extract per-cell NPZ for the Alps bbox (covers all selected cells).
    print(">>> Extracting features to cache ...")
    run_gfs_cache_creation(
        dates=f"{date_str}:{date_str}",
        bbox="6.0,43.0,17.0,49.0",
        source_dir=FORECAST_GRIB_DIR,
        output_dir=GFS_CACHE_DIR,
        force=True,
    )

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
    ts = pd.Timestamp(target)

    rows = []
    for cell in cells:
        lat, lon = map(int, cell.split("_"))
        rec = compute_day_features(cache, lat, lon, ts)
        if rec is None:
            continue
        x = np.array([[float(rec.get(f, 0.0)) for f in feature_names]], dtype=np.float32)
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
