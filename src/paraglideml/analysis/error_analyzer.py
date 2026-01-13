from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import EXPERIMENTS_DIR

# Feature units and categories
FEATURE_CATEGORIES = {
    "wind": [
        "wind_speed_10m",
        "wind_speed_850",
        "wind_speed_700",
        "wind_speed_500",
        "u_10m",
        "v_10m",
        "u_850",
        "v_850",
        "u_700",
        "v_700",
        "wind_shear_low",
        "ws_1000",
        "ws_975",
        "ws_950",
        "ws_925",
        "ws_900",
        "ws_850",
        "ws_800",
        "ws_750",
        "ws_700",
        "ws_600",
        "ws_500",
    ],
    "clouds": ["total_cloud_cover", "dps_850", "dps_700"],
    "stability": [
        "cape",
        "cin",
        "lapse_rate_low",
        "lapse_rate_mid",
        "lr_1000_975",
        "lr_975_950",
        "lr_950_925",
        "lr_925_900",
        "lr_900_850",
        "lr_850_800",
        "lr_800_750",
        "lr_750_700",
        "lr_700_600",
        "lr_600_500",
    ],
    "thermal": ["temp_2m", "surface_pressure"],
    "temporal": ["is_weekend", "day_of_year", "year"],
}

UNITS = {
    "surface_pressure": "Pa",
    "temp_2m": "K",
    "wind_speed_10m": "m/s",
    "wind_speed_850": "m/s",
    "wind_speed_700": "m/s",
    "wind_speed_500": "m/s",
    "wind_shear_low": "m/s",
    "u_10m": "m/s",
    "v_10m": "m/s",
    "u_850": "m/s",
    "v_850": "m/s",
    "u_700": "m/s",
    "v_700": "m/s",
    "dps_850": "C",
    "dps_700": "C",
    "cape": "J/kg",
    "cin": "J/kg",
    "total_cloud_cover": "%",
    "lapse_rate_low": "K/km",
    "lapse_rate_mid": "K/km",
}
for lvl in [
    "1000",
    "975",
    "950",
    "925",
    "900",
    "850",
    "800",
    "750",
    "700",
    "600",
    "500",
]:
    UNITS[f"ws_{lvl}"] = "m/s"
    UNITS[f"lr_{lvl}"] = "K/km"


@dataclass
class ExperimentSummary:
    name: str
    total_samples: int
    correct: int
    fn_count: int
    fp_count: int
    fn_rate: float
    fp_rate: float


def load_all_predictions(exp_name: str) -> Optional[Dict[str, pd.DataFrame]]:
    base_path = EXPERIMENTS_DIR / exp_name
    result = {}
    for category in ["tp", "tn", "fp", "fn", "neutral_zone", "all_predictions"]:
        path = base_path / f"{category}.csv"
        if path.exists():
            result[category] = pd.read_csv(path)
    return result if result else None


def compare_distributions(
    correct_df: pd.DataFrame, error_df: pd.DataFrame, features: List[str]
) -> Dict[str, Dict]:
    results = {}
    for feat in features:
        if feat not in correct_df.columns or feat not in error_df.columns:
            continue
        correct_vals = correct_df[feat].dropna()
        error_vals = error_df[feat].dropna()
        if len(correct_vals) < 2 or len(error_vals) < 2:
            continue
        mean_diff = error_vals.mean() - correct_vals.mean()
        pooled_std = np.sqrt((correct_vals.std() ** 2 + error_vals.std() ** 2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        results[feat] = {
            "correct_mean": correct_vals.mean(),
            "error_mean": error_vals.mean(),
            "mean_diff": mean_diff,
            "effect_size": effect_size,
            "unit": UNITS.get(feat, ""),
        }
    return results


def identify_influential_features(data: Dict[str, pd.DataFrame], top_n: int = 10) -> None:
    all_df = data.get("all_predictions")
    if all_df is None:
        return

    features = [
        c
        for c in all_df.columns
        if all_df[c].dtype in ["float64", "int64"]
        and c
        not in [
            "is_flyable",
            "pred",
            "prob",
            "is_correct",
            "in_neutral_zone",
            "flight_count",
            "year",
            "day_of_year",
            "is_weekend",
        ]
    ]

    correct = pd.concat([data.get("tp", pd.DataFrame()), data.get("tn", pd.DataFrame())])
    errors = pd.concat([data.get("fp", pd.DataFrame()), data.get("fn", pd.DataFrame())])

    if len(correct) == 0 or len(errors) == 0:
        return

    comparisons = compare_distributions(correct, errors, features)
    sorted_features = sorted(
        comparisons.items(), key=lambda x: abs(x[1]["effect_size"]), reverse=True
    )

    print("\n=== INFLUENTIAL FEATURES (ERRORS vs CORRECT) ===")
    print(f"{ 'Feature':<25} {'Direction':>10} {'Δ Mean':>10} {'Effect':>8}")
    print("-" * 60)

    for feat, stats in sorted_features[:top_n]:
        direction = "↑ HIGHER" if stats["effect_size"] > 0 else "↓ lower"
        color = "+" if stats["effect_size"] > 0 else ""
        abs_effect = abs(stats["effect_size"])
        magnitude = "LARGE" if abs_effect > 0.8 else "medium" if abs_effect > 0.5 else "small"
        unit = f" {stats['unit']}" if stats["unit"] else ""
        print(
            f"{feat:<25} {direction:>10} {color}{stats['mean_diff']:>9.2f}{unit:<5} {magnitude:>8}"
        )


def analyze_experiment(exp_name: Optional[str] = None, output_dir: Optional[Path] = None) -> None:
    if exp_name:
        exp_path = EXPERIMENTS_DIR / exp_name
    else:
        exp_dirs = sorted([d for d in EXPERIMENTS_DIR.glob("exp_*") if d.is_dir()])
        if not exp_dirs:
            print("Эксперименты не найдены")
            return
        exp_path = exp_dirs[-1]
        exp_name = exp_path.name

    data = load_all_predictions(exp_name)
    if data is None:
        print(f"Error: No data found for {exp_name} in {EXPERIMENTS_DIR}")
        return

    print(f"\n{'='*50}\nEXPERIMENT: {exp_name}\n{'='*50}")
    for cat in ["tp", "tn", "fp", "fn"]:
        df = data.get(cat, pd.DataFrame())
        print(f"  {cat.upper()}: {len(df)}")

    identify_influential_features(data)

    if output_dir and data.get("all_predictions") is not None:
        print(f"\nTimeline plots skipped in this CLI version. (Module: {__name__})")
