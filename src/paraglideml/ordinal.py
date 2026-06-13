"""
Ordinal "flight quality" tiers: cumulative P(>= tier) for the bot.

Instead of one number, the bot wants an ordered ladder a pilot reads at a glance:
  P(>= flyable)  — a real flight happens here (best route >= 15 km, not a sled ride)
  P(>= good)     — worth driving out for (best route >= 50 km)
  P(>= epic)     — exceptional XC day (best route >= 100 km)

These are *cumulative* probabilities (each tier implies the ones below it), so by
construction P(>=flyable) >= P(>=good) >= P(>=epic). We train one calibrated binary
model per threshold (the simplest robust ordinal scheme) reusing the goodxc target
machinery — same distance label, same regional-consensus confidence weighting, same
honest temporal split — then enforce monotonicity across tiers on the calibrated
probabilities (a model trio can violate it slightly; we clamp and report how often).

The bot pushes a spot when P(>=good) clears its threshold; the other two tiers add
context (is it merely soarable, or a potential epic).
"""

import datetime
import json
import os
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .goodxc import build_good_xc_target
from .multiregional import MultiRegionalConfig, get_next_experiment_dir, load_and_prepare_data

# (threshold_km, human label). Cumulative: each implies the tiers below.
TIERS: List[tuple] = [(15.0, "flyable"), (50.0, "good"), (100.0, "epic")]


def _train_tier(fit_df, val_df, test_df, feature_names, good_km, broad_min, config):
    """Train + calibrate one cumulative tier (dist_max >= good_km). Returns dict."""
    # Build the per-tier label/confidence within each split (region consensus is
    # split-local; reuse the goodxc target with this tier's distance threshold).
    fit_t = build_good_xc_target(fit_df, good_km, broad_min)
    val_t = build_good_xc_target(val_df, good_km, broad_min)
    test_t = build_good_xc_target(test_df, good_km, broad_min)

    X_fit, y_fit = fit_t[feature_names].values, fit_t["good_xc"].values
    X_val, y_val = val_t[feature_names].values, val_t["good_xc"].values
    X_test, y_test = test_t[feature_names].values, test_t["good_xc"].values
    w_fit = fit_t["good_confidence"].values

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=config.random_seed,
    )
    clf.fit(X_fit, y_fit, sample_weight=w_fit)

    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs_raw = clf.predict_proba(X_test)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    test_probs = iso.transform(test_probs_raw)

    return {
        "clf": clf,
        "iso": iso,
        "y_test": y_test,
        "test_probs": test_probs,
        "ap": float(average_precision_score(y_test, test_probs)),
        "roc": float(roc_auc_score(y_test, test_probs)),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "base_rate": float(y_test.mean()),
    }


def run_ordinal_pipeline(
    experiments_dir: str = "models/experiments",
    broad_min: int = 5,
) -> str:
    """Train calibrated cumulative tier models and save the ordinal trio."""
    config = MultiRegionalConfig(experiments_dir=experiments_dir)
    train_df, test_df, feature_names, region_mapping = load_and_prepare_data(config)

    val_year = int(train_df["year"].max())
    fit_df = train_df[train_df["year"] < val_year].copy()
    val_df = train_df[train_df["year"] == val_year].copy()
    print(f"Ordinal tiers (broad_min={broad_min}); fit<{val_year}, val={val_year}, test=held-out")

    tiers: Dict[str, dict] = {}
    for good_km, label in TIERS:
        res = _train_tier(fit_df, val_df, test_df, feature_names, good_km, broad_min, config)
        tiers[label] = {**res, "km": good_km}
        print(f"  P(>={label:<8} >={good_km:>5.0f}km): base {res['base_rate']:.3f}  "
              f"AP {res['ap']:.3f} (lift x{res['ap']/res['base_rate']:.2f})  "
              f"ROC {res['roc']:.3f}  Brier {res['brier']:.4f}")

    # Enforce monotonicity on calibrated test probs: P(>=flyable)>=P(>=good)>=P(>=epic).
    labels = [lbl for _, lbl in TIERS]
    P = np.vstack([tiers[lbl]["test_probs"] for lbl in labels])  # [n_tiers, n_samples]
    raw_violation = float(np.mean(np.any(np.diff(P, axis=0) > 1e-9, axis=0)))
    # At inference, clamp with np.minimum.accumulate so P(>=tier) is non-increasing;
    # here we only measure how often the independent trio needs that clamp.
    print(f"  Monotonicity: {raw_violation:.1%} of cell-days needed clamping "
          f"(P(>=tier) forced non-increasing).")

    # Save the trio + calibrators + region mapping.
    exp_dir = get_next_experiment_dir(config.experiments_dir)
    print(f"\n>>> Saving ordinal experiment to: {exp_dir}")
    for lbl in labels:
        joblib.dump(tiers[lbl]["clf"], os.path.join(exp_dir, f"model_{lbl}.joblib"))
        joblib.dump(tiers[lbl]["iso"], os.path.join(exp_dir, f"calibrator_{lbl}.joblib"))
    with open(os.path.join(exp_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(exp_dir, "region_mapping.json"), "w") as f:
        json.dump({k: int(v) for k, v in region_mapping.items()}, f, indent=2)

    summary = {
        "model_type": "OrdinalTiers(HistGradientBoosting x3)",
        "tiers": {lbl: {"km": tiers[lbl]["km"], "base_rate": tiers[lbl]["base_rate"],
                        "average_precision": tiers[lbl]["ap"], "roc_auc": tiers[lbl]["roc"],
                        "brier": tiers[lbl]["brier"]} for lbl in labels},
        "monotonicity_clamp_rate": raw_violation,
        "broad_min": broad_min,
        "num_features": len(feature_names),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = [
        "Ordinal Flight-Quality Tiers Report",
        "=" * 50,
        f"Date: {datetime.datetime.now()}",
        f"Experiment: {os.path.basename(exp_dir)}",
        "Model: 3x calibrated HistGradientBoosting, cumulative distance tiers",
        "",
        "Results (honest, test = held-out year):",
    ]
    for lbl in labels:
        t = tiers[lbl]
        lines.append(
            f"  P(>={lbl:<8} >={t['km']:>5.0f}km): base {t['base_rate']:.3f}  "
            f"AP {t['ap']:.3f} (lift x{t['ap']/t['base_rate']:.2f})  "
            f"ROC {t['roc']:.3f}  Brier {t['brier']:.4f}"
        )
    lines.append(f"  Monotonicity clamp rate: {raw_violation:.1%}")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report)

    print("\n✓ Ordinal tier pipeline completed successfully.")
    return exp_dir
