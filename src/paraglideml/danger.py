"""Dangerous-day ("blow-out" / overdevelopment) detector — a SEPARATE gate.

Motivation (see convective-features-experiment memory): the XC tier model is trained
on flight distance, so on a convectively loaded day it predicts high P(flyable) — even
when the day actually overdevelops into thunderstorms / blows out with too much wind and
NOBODY flies (the 2026-06-15 46_13 case: flyable ~90%, reality dangerous, 0 flights,
wind 7/12). You cannot fix this inside the distance target: instability is *rewarded*
there (it usually does produce big XC). "Dangerous" is a safety concept absent from the
distance label.

So this is a standalone detector with its OWN proxy label, meant to be shown as a ⚡
warning ALONGSIDE the tiers, not folded into them.

Proxy label (historically validated separators, AUC 0.71-0.79 single-feature):
  Operate only WITHIN the convectively active regime (cape_daymax >= CAPE_MIN) — that's
  where the XC model is false-positive-prone and where "overdevelopment" is even a
  question. There:
    blowout = 1 : region was loaded but DEAD — flight_count == 0 AND region_n_good == 0
                  (regional consensus denoises "nobody happened to fly" presence noise)
    blowout = 0 : region DELIVERED — good_xc == 1 (a real >=50 km XC flew in the cell)
    excluded    : the ambiguous middle (some local flights / partially active region)
  Among unstable days the dead ones are the overcast / moist-mid-level / windy-aloft /
  high-shear days — exactly overdevelopment & blow-out.

Features are WEATHER-ONLY (no flight_count / region_n_good / good_xc) so the detector
deploys on a pure forecast feature row, identical at train and serve.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .goodxc import build_good_xc_target

CAPE_MIN = 800.0  # convectively active regime: where overdevelopment is a real question

# Weather-only discriminators. Within the unstable regime, the dead-vs-delivered split is
# driven by mid-level moisture, cloud, wind aloft and shear (NOT the instability indices
# CAPE/TT/KI, which are high on both good and blown-out days).
DANGER_FEATURES: List[str] = [
    "mid_rh",                 # moist mid-levels feed overdevelopment (44% delivered vs 67% dead)
    "total_cloud_cover",      # overcast = no sun = dead (6% vs 69%)
    "cape_shear_daymax",      # CAPE x deep shear: organised/violent convection (12k vs 21k)
    "deep_shear",             # surface->500 hPa bulk shear
    "ws700_daymax",           # wind aloft (5.6 vs 13.7 m/s) — blow-out / strong gradient
    "ws850_daymax",
    "cin",                    # uncapped (CIN near 0) pops early -> overdevelops
    "dewpoint_spread_2m",     # low spread = low cloudbase / moist surface
    "omega_low_min",          # strong large-scale ascent = forced convection
    "total_totals_daymax",    # severe-storm index (context, weak separator)
    "k_index_daymax",         # thunderstorm/moisture index (context)
    "cape_daymax",            # regime context
]

ARTIFACT_FILES = ("danger_model.joblib", "danger_calibrator.joblib", "danger_features.txt", "danger_config.json")


def build_blowout_target(
    df: pd.DataFrame, cape_min: float = CAPE_MIN, broad_min: int = 5
) -> pd.DataFrame:
    """Add ``blowout`` (1 dead / 0 delivered / -1 excluded) and ``unstable`` mask.

    Regional consensus is computed within the passed frame (pass train/test separately so
    it never crosses the temporal split — each date lives in one split, leakage-safe).
    """
    df = build_good_xc_target(df, 50.0, broad_min)  # adds good_xc, region_n_good
    unstable = df["cape_daymax"].values >= cape_min
    dead = unstable & (df["flight_count"].values == 0) & (df["region_n_good"].values == 0)
    delivered = unstable & (df["good_xc"].values == 1)
    blow = np.full(len(df), -1, dtype=int)
    blow[delivered] = 0
    blow[dead] = 1
    df["blowout"] = blow
    df["unstable"] = unstable
    return df


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["blowout"] >= 0]


def rule_risk(rec: Dict[str, float]) -> int:
    """Transparent fallback rule (no training): a coarse overdevelopment flag.

    High mid-level moisture AND substantial cloud, with either violent shear-energy or
    strong wind aloft. Returned as 0/1; the trained detector is the primary score.
    """
    moist = rec.get("mid_rh", 0.0) >= 55.0
    cloudy = rec.get("total_cloud_cover", 0.0) >= 50.0
    energetic = rec.get("cape_shear_daymax", 0.0) >= 15000.0 or rec.get("ws700_daymax", 0.0) >= 10.0
    return int(moist and cloudy and energetic)


def _fit(train_lab: pd.DataFrame, features: List[str], seed: int) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    clf.fit(train_lab[features].values, train_lab["blowout"].values)
    return clf


def run_danger_backtest(
    cape_min: float = CAPE_MIN, broad_min: int = 5, features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Rolling-origin backtest: fit on all prior years' labelled unstable days, score on
    the year. Reports ROC-AUC / AP (vs base rate) and precision/recall of the ⚡ flag at a
    0.5 threshold, plus the transparent rule's precision/recall, per test year."""
    from .multiregional import MultiRegionalConfig, cluster_regions, resolve_num_regions

    feats = features or DANGER_FEATURES
    cfg = MultiRegionalConfig()
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    n_regions = resolve_num_regions(int(df["cell_id"].nunique()), floor=cfg.num_regions)
    rm = cluster_regions(df, n_clusters=n_regions, random_state=cfg.random_seed)
    df["region_id"] = df["cell_id"].map(rm)
    years = sorted(int(y) for y in df["year"].unique())
    print(f"Danger detector backtest over {years}  ({len(feats)} weather features, cape_min={cape_min})")

    rows = []
    for Y in years[1:]:
        tr = build_blowout_target(df[df["year"] < Y], cape_min, broad_min)
        te = build_blowout_target(df[df["year"] == Y], cape_min, broad_min)
        tr_l, te_l = _labelled(tr), _labelled(te)
        y = te_l["blowout"].values
        if len(tr_l) < 200 or not (0 < int(y.sum()) < len(y)):
            continue
        clf = _fit(tr_l, feats, cfg.random_seed)
        p = clf.predict_proba(te_l[feats].values)[:, 1]
        pred = (p > 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        rule = te_l.apply(lambda r: rule_risk(r.to_dict()), axis=1).values
        rpr, rrc, _, _ = precision_recall_fscore_support(y, rule, average="binary", zero_division=0)
        rows.append({
            "year": Y, "n_unstable_labelled": int(len(te_l)), "base_blowout": float(y.mean()),
            "roc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "model_P": float(pr), "model_R": float(rc), "model_F1": float(f1),
            "rule_P": float(rpr), "rule_R": float(rrc),
        })

    res = pd.DataFrame(rows)
    print("\n=== Danger detector: rolling backtest (blowout = unstable-but-dead) ===")
    print(f"{'year':>5} {'n':>5} {'base':>6} {'ROC':>6} {'AP':>6} | {'modelP':>7}{'modelR':>7}{'F1':>6} | {'ruleP':>6}{'ruleR':>6}")
    for _, r in res.iterrows():
        print(f"{int(r.year):>5} {int(r.n_unstable_labelled):>5} {r.base_blowout:>6.2f} "
              f"{r.roc:>6.3f} {r.ap:>6.3f} | {r.model_P:>7.2f}{r.model_R:>7.2f}{r.model_F1:>6.2f} | "
              f"{r.rule_P:>6.2f}{r.rule_R:>6.2f}")
    if len(res):
        print(f"{'mean':>5} {'':>5} {res.base_blowout.mean():>6.2f} {res.roc.mean():>6.3f} "
              f"{res.ap.mean():>6.3f} | {res.model_P.mean():>7.2f}{res.model_R.mean():>7.2f}"
              f"{res.model_F1.mean():>6.2f} | {res.rule_P.mean():>6.2f}{res.rule_R.mean():>6.2f}")
    return res


def train_danger_detector(
    out_dir: str, cape_min: float = CAPE_MIN, broad_min: int = 5,
    features: Optional[List[str]] = None, calib_year: Optional[int] = None,
) -> str:
    """Fit the detector on ALL labelled unstable days, isotonic-calibrate on the last year,
    and save artifacts (model + calibrator + features + config) to ``out_dir``."""
    from .multiregional import MultiRegionalConfig, cluster_regions, resolve_num_regions

    feats = features or DANGER_FEATURES
    cfg = MultiRegionalConfig()
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    n_regions = resolve_num_regions(int(df["cell_id"].nunique()), floor=cfg.num_regions)
    rm = cluster_regions(df, n_clusters=n_regions, random_state=cfg.random_seed)
    df["region_id"] = df["cell_id"].map(rm)
    df = build_blowout_target(df, cape_min, broad_min)
    lab = _labelled(df)

    cy = calib_year if calib_year is not None else int(lab["year"].max())
    fit = lab[lab["year"] < cy] if (lab["year"] < cy).sum() >= 200 else lab
    clf = _fit(fit, feats, cfg.random_seed)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal_src = lab[lab["year"] == cy]
    if len(cal_src) >= 30:
        cal.fit(clf.predict_proba(cal_src[feats].values)[:, 1], cal_src["blowout"].values)
    else:
        cal = None

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "danger_model.joblib"))
    if cal is not None:
        joblib.dump(cal, os.path.join(out_dir, "danger_calibrator.joblib"))
    with open(os.path.join(out_dir, "danger_features.txt"), "w") as f:
        f.write("\n".join(feats))
    with open(os.path.join(out_dir, "danger_config.json"), "w") as f:
        json.dump({"cape_min": cape_min, "broad_min": broad_min, "n_features": len(feats),
                   "n_train": int(len(fit)), "blowout_base_rate": float(lab["blowout"].mean()),
                   "calibrated": cal is not None, "calib_year": cy}, f, indent=2)
    print(f">>> Saved danger detector ({len(feats)} feats, {len(fit)} train rows) to {out_dir}")
    return out_dir


def load_danger_detector(model_dir: str):
    feats = [l for l in open(os.path.join(model_dir, "danger_features.txt")).read().splitlines() if l.strip()]
    clf = joblib.load(os.path.join(model_dir, "danger_model.joblib"))
    cal_path = os.path.join(model_dir, "danger_calibrator.joblib")
    cal = joblib.load(cal_path) if os.path.exists(cal_path) else None
    return feats, clf, cal


def overdevelopment_risk(rec: Dict[str, float], feats: List[str], clf, cal=None) -> float:
    """P(blow-out / dangerous overdevelopment) for one cell-day weather record [0,1].

    Only meaningful in the convectively active regime; on calm days CAPE is low and the
    XC model already reads low — apply/show the flag where cape_daymax >= CAPE_MIN.
    """
    x = np.array([[float(rec.get(f, 0.0)) for f in feats]])
    p = clf.predict_proba(x)[:, 1]
    if cal is not None:
        p = cal.transform(p)
    return float(p[0])
