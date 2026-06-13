"""
Forecast-skew measurement: how much does AP drop when the ordinal model is fed GFS
*forecast* lead-times (f024..f120) instead of the *analysis* (f000) it trained on?

The model learned [analysis weather f000] -> [flight outcome]. In production the bot
must use a real forecast: today's 00z run, valid +1..+5 days out. Those forecast fields
are smoother / biased relative to the analysis, so the model sees slightly
out-of-distribution inputs. This module quantifies the cost so we can decide v1:
ship-as-is on short horizons (small ΔAP) vs forecast-aware retrain (large ΔAP).

Method (apples-to-apples). Take a window of recent cell-days with KNOWN outcomes.
  - Baseline = the analysis features already in multicell_dataset.csv (f000).
  - For each lead L (in days): re-download those SAME valid days from the run issued
    L days earlier (00z run, forecast hour fxx = L*24 + {6,12,18}), extract features
    identically (compute_day_features + spot terrain), score with the SAME exp_056 trio.
Labels (dist_max -> tier) are identical for both; only the weather input differs. For
each lead we restrict baseline AND forecast to the exact same cell-days that the
forecast extraction covered, so ΔAP(L) = AP_forecast(L) - AP_analysis is honest.

See [[flybeeper-integration-goal]] (the deployment train/serve question this answers).
"""

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import GFS_FORECAST_DIR, PROCESSED_DATA_DIR
from .data.dataset_builder import compute_day_features
from .data.gfs_processor import run_gfs_cache_creation
from .data.terrain import add_terrain_features, load_cell_terrain
from .data.weather_cache import WeatherCache
from .predict import _resolve_ordinal_experiment, download_gfs_slice
from .tiers import TIERS

BBOX = "6.0,43.0,17.0,49.0"
FORECAST_HOURS = (6, 12, 18)


def _forecast_day_grib(target: dt.date, lead_days: int, grib_root: Path) -> Dict[int, bool]:
    """Download the 06/12/18z slices for `target` from the run issued `lead_days` earlier.

    run = target - lead_days at 00z; forecast hour fxx = lead_days*24 + valid_hour. Files
    are named with the VALID day/hour so the existing extractor (which keys on valid time)
    picks them up unchanged. Resumable: existing non-empty files are kept.
    """
    run_date = target - dt.timedelta(days=lead_days)
    out: Dict[int, bool] = {}
    for h in FORECAST_HOURS:
        dest = grib_root / target.strftime("%Y-%m") / f"gfsanl_3_{target:%Y%m%d}_{h:02d}00_000.grb2"
        if dest.exists() and dest.stat().st_size > 0:
            out[h] = True
            continue
        fxx = lead_days * 24 + h
        out[h] = download_gfs_slice(run_date, 0, dest, fxx=fxx)
    return out


def _vectors_from_cache(
    cache_root: Path,
    cells: List[str],
    dates: List[pd.Timestamp],
    feature_names: List[str],
    terrain: dict,
) -> Tuple[List[Tuple[str, pd.Timestamp]], np.ndarray]:
    """Build (key, feature-vector) pairs from a cache, same code path as inference."""
    cache = WeatherCache(cache_root=str(cache_root))
    keys: List[Tuple[str, pd.Timestamp]] = []
    rows: List[List[float]] = []
    for cell in cells:
        lat, lon = map(int, cell.split("_"))
        tcell = terrain.get(cell)
        for d in dates:
            rec = compute_day_features(cache, lat, lon, d)
            if rec is None:
                continue
            rec = add_terrain_features(dict(rec), tcell)
            rows.append([float(rec.get(f, 0.0)) for f in feature_names])
            keys.append((cell, pd.Timestamp(d).normalize()))
    X = (
        np.array(rows, dtype=np.float32)
        if rows
        else np.empty((0, len(feature_names)), dtype=np.float32)
    )
    return keys, X


def _score_tier(models, calibrators, lbl, X, y) -> Optional[dict]:
    """Calibrated AP / ROC / Brier for one tier (None if degenerate)."""
    if X.shape[0] == 0 or y.sum() == 0 or y.sum() == len(y):
        return None
    raw = models[lbl].predict_proba(X)[:, 1]
    p = calibrators[lbl].transform(raw)
    return {
        "n": int(len(y)),
        "base": float(y.mean()),
        "ap": float(average_precision_score(y, p)),
        "roc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def run_forecast_skew(
    start_date: str,
    end_date: str,
    leads: Sequence[int] = (1, 3, 5),
    experiment: Optional[str] = None,
    sample_every: int = 1,
    selected_cells_path: Optional[Path] = None,
    scratch_root: Optional[str] = None,
) -> pd.DataFrame:
    """Measure AP/ROC degradation of the ordinal model on forecast vs analysis inputs.

    For each lead (days ahead) prints a tier x lead table of AP/ROC and Δ vs analysis,
    computed on the exact same cell-days the forecast covered. Returns the long DataFrame.
    """
    exp_dir = _resolve_ordinal_experiment(experiment)
    feature_names = [l for l in (exp_dir / "features.txt").read_text().splitlines() if l.strip()]
    tier_labels = [lbl for _, lbl in TIERS]
    models = {lbl: joblib.load(exp_dir / f"model_{lbl}.joblib") for lbl in tier_labels}
    calibrators = {lbl: joblib.load(exp_dir / f"calibrator_{lbl}.joblib") for lbl in tier_labels}
    terrain = load_cell_terrain() or {}

    if selected_cells_path is None:
        selected_cells_path = PROCESSED_DATA_DIR / "selected_cells.json"
    cells = json.loads(Path(selected_cells_path).read_text())

    # Window + labels + baseline (analysis) features, all straight from the dataset CSV.
    df = pd.read_csv(PROCESSED_DATA_DIR / "multicell_dataset.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    win = df[(df["date"] >= lo) & (df["date"] <= hi)].copy()
    all_dates = sorted(win["date"].unique())
    dates = [pd.Timestamp(d) for d in all_dates[::sample_every]]
    win = win[win["date"].isin(dates)].copy()
    win_idx = win.set_index(["cell_id", "date"]).sort_index()
    dist_by_key = win_idx["dist_max"].to_dict()

    print(f">>> Forecast-skew vs {exp_dir.name}: leads {list(leads)} d, "
          f"window {start_date}..{end_date} every {sample_every}d "
          f"({len(dates)} days x {len(cells)} cells = {len(win)} baseline cell-days)")

    def baseline_on(keys: List[Tuple[str, pd.Timestamp]]):
        present = [k for k in keys if k in dist_by_key]
        Xb = win_idx.loc[present, feature_names].to_numpy(dtype=np.float32)
        return present, Xb

    rows = []
    for km, lbl in TIERS:
        keys = list(dist_by_key.keys())
        present, Xb = baseline_on(keys)
        y = np.array([1 if dist_by_key[k] >= km else 0 for k in present])
        s = _score_tier(models, calibrators, lbl, Xb, y)
        rows.append({"lead": 0, "tier": lbl, "km": km, **(s or {})})

    root = Path(scratch_root) if scratch_root else (GFS_FORECAST_DIR / "skew")
    for L in leads:
        grib_root = root / f"lead{L}d_grib"
        cache_root = root / f"lead{L}d_cache"
        print(f"\n--- lead {L} day(s) ---")
        n12 = sum(1 for d in dates if _forecast_day_grib(d.date(), L, grib_root).get(12))
        print(f"  {n12}/{len(dates)} target days have a 12z forecast slice; extracting ...")
        ds = ",".join(f"{d:%Y-%m-%d}:{d:%Y-%m-%d}" for d in dates)
        run_gfs_cache_creation(dates=ds, bbox=BBOX, source_dir=grib_root,
                               output_dir=cache_root, force=False)
        fkeys, Xf = _vectors_from_cache(cache_root, cells, dates, feature_names, terrain)
        common = [k for k in fkeys if k in dist_by_key]
        idx_keep = [i for i, k in enumerate(fkeys) if k in dist_by_key]
        Xf = Xf[idx_keep] if len(idx_keep) else Xf
        # Baseline restricted to the SAME covered cell-days (apples-to-apples ΔAP).
        _, Xb = baseline_on(common)
        for km, lbl in TIERS:
            y = np.array([1 if dist_by_key[k] >= km else 0 for k in common])
            f = _score_tier(models, calibrators, lbl, Xf, y)
            b = _score_tier(models, calibrators, lbl, Xb, y)
            row = {"lead": L, "tier": lbl, "km": km, **(f or {})}
            if f and b:
                row["d_ap"] = f["ap"] - b["ap"]
                row["d_roc"] = f["roc"] - b["roc"]
                row["base_ap"] = b["ap"]
                row["base_roc"] = b["roc"]
            rows.append(row)

    res = pd.DataFrame(rows)
    _print_skew_table(res, leads)
    return res


def _print_skew_table(res: pd.DataFrame, leads: Sequence[int]) -> None:
    """Tier x lead matrix of AP and ROC with Δ vs analysis."""
    print("\n" + "=" * 70)
    print("FORECAST SKEW — AP by tier x lead (Δ vs analysis on same cell-days)")
    print("=" * 70)
    head = f"  {'tier':<8} {'base':>5}  {'anl(f000)':>9}"
    for L in leads:
        head += f"   {'+'+str(L)+'d':>13}"
    print(head)
    for _, lbl in TIERS:
        anl = res[(res.lead == 0) & (res.tier == lbl)].iloc[0]
        line = f"  {lbl:<8} {anl.get('base', float('nan')):>5.3f}  {anl.get('ap', float('nan')):>9.3f}"
        for L in leads:
            r = res[(res.lead == L) & (res.tier == lbl)]
            if len(r) and not pd.isna(r.iloc[0].get("ap", np.nan)):
                rr = r.iloc[0]
                line += f"   {rr['ap']:>6.3f}({rr.get('d_ap', float('nan')):+.3f})"
            else:
                line += f"   {'n/a':>13}"
        print(line)
    print("\n  ROC-AUC:")
    for _, lbl in TIERS:
        anl = res[(res.lead == 0) & (res.tier == lbl)].iloc[0]
        line = f"  {lbl:<8} {'':>5}  {anl.get('roc', float('nan')):>9.3f}"
        for L in leads:
            r = res[(res.lead == L) & (res.tier == lbl)]
            if len(r) and not pd.isna(r.iloc[0].get("roc", np.nan)):
                rr = r.iloc[0]
                line += f"   {rr['roc']:>6.3f}({rr.get('d_roc', float('nan')):+.3f})"
            else:
                line += f"   {'n/a':>13}"
        print(line)
    print("\n  Δ = forecast − analysis (negative = forecast worse). Small Δ on short")
    print("  leads ⇒ v1 ship-as-is is fine; large Δ ⇒ forecast-aware retrain warranted.")
