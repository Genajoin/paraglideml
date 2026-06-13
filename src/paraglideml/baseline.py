"""
Gradient-boosted baseline (sklearn HistGradientBoostingClassifier).

A tabular gradient-boosting model on the SAME features, labels and honest
evaluation protocol as the neural MultiRegionalModel:
  - temporal train/val split (val = most recent training year),
  - decision threshold chosen on validation (never on test),
  - reported on the held-out test year.

Gradient-boosted trees are a strong baseline for structured weather data and
often match or beat a small MLP while being cheaper to train and trivial to
ship. This command exists to establish the realistic ceiling before investing
in deeper architectures, and as a candidate model for the FlyBeeper pipeline.
"""

import datetime
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score

from .multiregional import (
    MultiRegionalConfig,
    find_optimal_threshold,
    get_next_experiment_dir,
    load_and_prepare_data,
)


def run_baseline_pipeline(
    experiments_dir: str = "models/experiments",
    learning_rate: float = 0.05,
    max_iter: int = 400,
    max_leaf_nodes: int = 31,
    use_confidence_weighting: bool = True,
) -> str:
    """Train and evaluate a gradient-boosted baseline with the honest protocol."""
    config = MultiRegionalConfig(experiments_dir=experiments_dir)

    # Reuse the exact same loading / feature selection / year split as the NN.
    train_df, test_df, feature_names, _ = load_and_prepare_data(config)

    # Temporal validation split: hold out the most recent training year.
    val_year = int(train_df["year"].max())
    fit_df = train_df[train_df["year"] < val_year]
    val_df = train_df[train_df["year"] == val_year]

    X_fit = fit_df[feature_names].values
    y_fit = fit_df["is_flyable"].values
    X_val = val_df[feature_names].values
    y_val = val_df["is_flyable"].values
    X_test = test_df[feature_names].values
    y_test = test_df["is_flyable"].values

    sample_weight = fit_df["label_confidence"].values if use_confidence_weighting else None

    print(f"Gradient-boosted baseline: {len(feature_names)} features")
    print(f"  Train: {len(X_fit)} (years < {val_year})  Val: {len(X_val)} ({val_year})  Test: {len(X_test)}")

    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=config.random_seed,
    )
    clf.fit(X_fit, y_fit, sample_weight=sample_weight)

    # Threshold on validation, freeze, apply once to test.
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_threshold, val_macro_f1, _ = find_optimal_threshold(y_val, val_probs)
    print(f"Threshold (chosen on val): {best_threshold:.2f}  |  Val Macro F1: {val_macro_f1:.3f}")

    test_probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (test_probs > best_threshold).astype(int)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_macro_f1_at_05 = f1_score(y_test, (test_probs > 0.5).astype(int), average="macro")
    print(f"Test Macro F1 @{best_threshold:.2f}: {test_macro_f1:.3f}  |  @0.50: {test_macro_f1_at_05:.3f}")

    report_metrics = classification_report(y_test, y_pred, target_names=["Not Flyable", "Flyable"])

    # Permutation importance on validation: which features actually drive the model.
    print("\n>>> Computing permutation importance (val)...")
    perm = permutation_importance(
        clf, X_val, y_val, n_repeats=5, random_state=config.random_seed, scoring="f1_macro"
    )
    order = np.argsort(perm.importances_mean)[::-1]
    top_features = [(feature_names[i], float(perm.importances_mean[i])) for i in order[:15]]
    top_str = "\n".join(f"  {name:24s} {imp:+.4f}" for name, imp in top_features)
    print(top_str)

    # Save experiment artifacts (comparable layout to the NN experiments).
    exp_dir = get_next_experiment_dir(config.experiments_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n>>> Saving baseline experiment to: {exp_dir}")

    joblib.dump(clf, os.path.join(exp_dir, "model.joblib"))
    with open(os.path.join(exp_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    summary = {
        "model_type": "HistGradientBoostingClassifier",
        "macro_f1": float(test_macro_f1),
        "macro_f1_at_0.5": float(test_macro_f1_at_05),
        "val_macro_f1": float(val_macro_f1),
        "threshold": float(best_threshold),
        "threshold_source": "validation (temporal)",
        "num_features": len(feature_names),
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "n_iter_": int(clf.n_iter_),
        "use_confidence_weighting": use_confidence_weighting,
        "top_features": top_features,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report_text = f"""
Gradient-Boosted Baseline Report
{'='*50}
Date: {datetime.datetime.now()}
Experiment: {os.path.basename(exp_dir)}
Model: HistGradientBoostingClassifier ({clf.n_iter_} trees)

Results (honest, threshold chosen on validation):
  Test Macro F1 @threshold: {test_macro_f1:.3f}
  Test Macro F1 @0.50:      {test_macro_f1_at_05:.3f}
  Val  Macro F1 @threshold: {val_macro_f1:.3f}
  Threshold (from val):     {best_threshold:.2f}
  Features:                 {len(feature_names)}

{report_metrics}

Top features (permutation importance, val, f1_macro):
{top_str}
"""
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report_text)

    print("\n✓ Baseline pipeline completed successfully.")
    return exp_dir
