"""
Distance-based "good XC day" target (the product target for CanFlyBot).

The binary ``is_flyable = (flight_count >= 1)`` label conflates a 200 km epic with
a 10 min sled ride: ~37% of "flyable" cell-days are local flights (<30 km). The
bot's real question is "is it worth driving out for a long cross-country flight?",
i.e. P(good XC day). We define that target from route ``distance`` and evaluate it
with precision-oriented, threshold-free metrics (Average Precision, calibration),
because the bot pushes only above a high-precision threshold (anti-noise).

Target construction (see target-methodology memory):
  - good_xc = (dist_max >= GOOD_KM)            # a real XC route flew here that day
  - confidence via REGIONAL consensus ("cut the middle"):
      confident Good : good_xc==1 AND region broadly good (>= BROAD_MIN cells)  -> 1.0
      lone   Good    : good_xc==1 but region not broad (single strong pilot?)   -> 0.5
      confident Bad  : flight_count==0 AND region quiet (no good cell)           -> 1.0
      ambiguous mid  : everything else (local-only days, empty cell on good day) -> MID_W

The middle is down-weighted (soft cut), not deleted, so the model still sees it but
trusts it less. ``--drop-middle`` hard-excludes it instead.

This module trains a gradient-boosted model (fast, strong on tabular weather) on the
new target with the same honest temporal protocol as the NN/baseline, and prints the
Average Precision of ``good_xc`` next to the AP of the old ``is_flyable`` label on the
same features/split — the headline measure of whether distance is a cleaner,
more weather-predictable target.
"""

import datetime
import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .multiregional import (
    MultiRegionalConfig,
    get_next_experiment_dir,
    load_and_prepare_data,
)

GOOD_KM = 50.0  # a "good XC day": best route in the cell reached this distance
EPIC_KM = 100.0
LOCAL_KM = 15.0
BROAD_MIN = 5  # region is "broadly good" if >= this many cells hit GOOD_KM that day
MID_W = 0.3  # weight of the ambiguous middle (soft cut)


def build_good_xc_target(
    df: pd.DataFrame, good_km: float = GOOD_KM, broad_min: int = BROAD_MIN
) -> pd.DataFrame:
    """
    Add ``good_xc`` label and ``good_confidence`` weight to a dataframe.

    Regional context (n cells reaching good_km per (region_id, date)) is computed
    WITHIN the dataframe; callers must pass train and test separately so the
    regional consensus never mixes across the temporal split (each date lives in
    exactly one split, so this is leakage-safe).
    """
    df = df.copy()
    df["good_xc"] = (df["dist_max"] >= good_km).astype(int)

    grp = df.groupby(["region_id", "date"])
    df["region_n_good"] = grp["good_xc"].transform("sum")

    conf = np.full(len(df), MID_W, dtype=np.float32)
    is_good = df["good_xc"].values == 1
    broad = df["region_n_good"].values >= broad_min
    quiet_region = df["region_n_good"].values == 0
    no_flight = df["flight_count"].values == 0

    conf[is_good & broad] = 1.0  # confident good
    conf[is_good & ~broad] = 0.5  # lone good
    conf[no_flight & quiet_region] = 1.0  # confident bad
    df["good_confidence"] = conf
    return df


def _metrics_at_threshold(y: np.ndarray, probs: np.ndarray, thr: float) -> Tuple[float, float, float]:
    pred = (probs > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def _threshold_for_precision(
    y_val: np.ndarray, probs_val: np.ndarray, target_precision: float = 0.80
) -> float:
    """Lowest threshold on val that reaches target precision (max recall at that P)."""
    grid = np.arange(0.05, 0.96, 0.01)
    best_thr, best_rec = None, -1.0
    for t in grid:
        p, r, _ = _metrics_at_threshold(y_val, probs_val, t)
        if p >= target_precision and r > best_rec:
            best_rec, best_thr = r, t
    if best_thr is None:  # never reaches target precision -> use most precise point
        best_thr = float(grid[np.argmax([_metrics_at_threshold(y_val, probs_val, t)[0] for t in grid])])
    return float(best_thr)


def run_goodxc_pipeline(
    experiments_dir: str = "models/experiments",
    good_km: float = GOOD_KM,
    broad_min: int = BROAD_MIN,
    learning_rate: float = 0.05,
    max_iter: int = 400,
    max_leaf_nodes: int = 31,
    drop_middle: bool = False,
    target_precision: float = 0.80,
) -> str:
    """Train a gradient-boosted P(good XC day) model with the honest protocol."""
    config = MultiRegionalConfig(experiments_dir=experiments_dir)
    train_df, test_df, feature_names, region_mapping = load_and_prepare_data(config)

    # Build the distance target within each split (region consensus is split-local).
    train_df = build_good_xc_target(train_df, good_km, broad_min)
    test_df = build_good_xc_target(test_df, good_km, broad_min)

    val_year = int(train_df["year"].max())
    fit_df = train_df[train_df["year"] < val_year].copy()
    val_df = train_df[train_df["year"] == val_year].copy()

    print(f"P(good XC) target: good_km={good_km}, broad_min={broad_min}, drop_middle={drop_middle}")
    print(f"  good_xc base rate -> fit:{fit_df['good_xc'].mean():.1%} "
          f"val:{val_df['good_xc'].mean():.1%} test:{test_df['good_xc'].mean():.1%}")

    if drop_middle:
        keep = fit_df["good_confidence"] >= 0.5
        print(f"  cutting the middle: keep {keep.mean():.1%} of train rows")
        fit_df = fit_df[keep]

    X_fit = fit_df[feature_names].values
    y_fit = fit_df["good_xc"].values
    X_val = val_df[feature_names].values
    y_val = val_df["good_xc"].values
    X_test = test_df[feature_names].values
    y_test = test_df["good_xc"].values
    w_fit = fit_df["good_confidence"].values

    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=config.random_seed,
    )
    clf.fit(X_fit, y_fit, sample_weight=w_fit)

    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    # --- Threshold-free quality: Average Precision (area under PR curve). ---
    ap_test = average_precision_score(y_test, test_probs)
    roc_test = roc_auc_score(y_test, test_probs)

    # Reference: same features/split predicting the OLD binary is_flyable label.
    # If good_xc is more weather-driven, its AP (vs its base rate) should be the
    # cleaner, more learnable signal.
    ref_clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=config.random_seed,
    )
    ref_clf.fit(fit_df[feature_names].values, fit_df["is_flyable"].values)
    ref_probs = ref_clf.predict_proba(X_test)[:, 1]
    ap_flyable = average_precision_score(test_df["is_flyable"].values, ref_probs)
    roc_flyable = roc_auc_score(test_df["is_flyable"].values, ref_probs)
    base_good = float(y_test.mean())
    base_flyable = float(test_df["is_flyable"].mean())

    # --- Operating points chosen on validation, applied once to test. ---
    thr_p = _threshold_for_precision(y_val, val_probs, target_precision)
    p_p, r_p, f1_p = _metrics_at_threshold(y_test, test_probs, thr_p)

    # --- Calibration: fit isotonic on val, measure Brier on test. ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    test_probs_cal = iso.transform(test_probs)
    brier_raw = brier_score_loss(y_test, test_probs)
    brier_cal = brier_score_loss(y_test, test_probs_cal)

    print("\n=== P(good XC day) — test (held-out year) ===")
    print(f"  Average Precision : {ap_test:.3f}   (base rate {base_good:.3f}, lift x{ap_test/base_good:.2f})")
    print(f"  ROC-AUC           : {roc_test:.3f}")
    print(f"  Reference is_flyable AP: {ap_flyable:.3f} (base {base_flyable:.3f}, lift x{ap_flyable/base_flyable:.2f}), ROC-AUC {roc_flyable:.3f}")
    print(f"  Anti-noise op point (val P>={target_precision:.0%}) thr={thr_p:.2f} -> "
          f"test precision {p_p:.2f}, recall {r_p:.2f}, F1 {f1_p:.2f}")
    print(f"  Brier: raw {brier_raw:.4f} -> isotonic-calibrated {brier_cal:.4f}")

    # Save experiment artifacts.
    exp_dir = get_next_experiment_dir(config.experiments_dir)
    print(f"\n>>> Saving good-xc experiment to: {exp_dir}")
    joblib.dump(clf, os.path.join(exp_dir, "model.joblib"))
    joblib.dump(iso, os.path.join(exp_dir, "calibrator.joblib"))
    with open(os.path.join(exp_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(exp_dir, "region_mapping.json"), "w") as f:
        json.dump({k: int(v) for k, v in region_mapping.items()}, f, indent=2)

    summary = {
        "model_type": "HistGradientBoostingClassifier",
        "target": "good_xc",
        "good_km": good_km,
        "epic_km": EPIC_KM,
        "broad_min": broad_min,
        "drop_middle": drop_middle,
        "threshold": float(thr_p),
        "threshold_source": f"validation (precision>={target_precision})",
        "average_precision": float(ap_test),
        "roc_auc": float(roc_test),
        "ap_base_rate": base_good,
        "ref_is_flyable_ap": float(ap_flyable),
        "ref_is_flyable_base_rate": base_flyable,
        "test_precision": p_p,
        "test_recall": r_p,
        "test_f1": f1_p,
        "brier_raw": float(brier_raw),
        "brier_calibrated": float(brier_cal),
        "num_features": len(feature_names),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report = f"""
P(good XC day) Report  —  distance target
{'='*50}
Date: {datetime.datetime.now()}
Experiment: {os.path.basename(exp_dir)}
Model: HistGradientBoostingClassifier ({clf.n_iter_} trees)
Target: good_xc = (dist_max >= {good_km} km), region broad_min={broad_min}, drop_middle={drop_middle}

Results (honest, test = held-out year):
  Average Precision (good_xc) : {ap_test:.3f}   base rate {base_good:.3f}  (lift x{ap_test/base_good:.2f})
  ROC-AUC                     : {roc_test:.3f}
  Reference is_flyable AP     : {ap_flyable:.3f}  base rate {base_flyable:.3f}  (lift x{ap_flyable/base_flyable:.2f})
  Anti-noise op point         : thr={thr_p:.2f} (val P>={target_precision:.0%})
      test precision={p_p:.2f}  recall={r_p:.2f}  F1={f1_p:.2f}
  Brier (raw -> calibrated)   : {brier_raw:.4f} -> {brier_cal:.4f}
  Features                    : {len(feature_names)}
"""
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report)

    print("\n✓ good-xc pipeline completed successfully.")
    return exp_dir
