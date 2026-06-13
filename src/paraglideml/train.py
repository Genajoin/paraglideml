import datetime
import os

import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .multiregional import (
    ExperimentSaver,
    MultiRegionalConfig,
    MultiRegionalModel,
    MultiRegionalTrainer,
    create_data_loaders,
    create_weighted_bce_loss,
    evaluate_by_regions,
    evaluate_per_cell,
    find_optimal_threshold,
    get_next_experiment_dir,
    load_and_prepare_data,
    plot_confusion_matrix,
    plot_per_region_performance,
    plot_threshold_analysis,
    plot_training_history,
)


def run_training_pipeline(
    num_regions: int = 3,
    epochs: int = 150,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    experiments_dir: str = "models/experiments",
):
    """
    Run the full training pipeline, reproducing the logic from the notebook.
    """
    # 1. Setup Configuration
    config = MultiRegionalConfig(
        num_regions=num_regions,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        experiments_dir=experiments_dir,
    )

    # 2. Load Data
    train_df, test_df, feature_names, region_mapping = load_and_prepare_data(config)

    # 3. Create Data Loaders (temporal val split, no test leakage)
    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        y_train,
        X_val,
        y_val,
        region_val,
        X_test,
        y_test,
        region_test,
    ) = create_data_loaders(train_df, test_df, feature_names, config)

    # 4. Initialize Model
    model = MultiRegionalModel(
        input_dim=len(feature_names),
        num_regions=config.num_regions,
        embedding_dim=config.regional_embedding_dim,
        dropout_rate=config.dropout_rate,
    )

    # 5. Setup Training
    # pos_weight=1.0: the dataset is near-balanced (~53% flyable), so the previous
    # auto pos_weight (num_neg/num_pos < 1) only down-weighted the positive class
    # and pushed the decision threshold up. Keep class weighting neutral and let
    # confidence weighting do its separate job.
    criterion = create_weighted_bce_loss(
        y_train, pos_weight=1.0, use_confidence=config.use_confidence_weighting
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    trainer = MultiRegionalTrainer(model, config, criterion, optimizer)

    # 6. Train
    print("\n>>> Starting Model Training...")
    trainer.fit(train_loader, val_loader, verbose=True)

    # 7. Select the decision threshold on the temporal validation set (year 2024),
    #    NOT on the test set. Picking the threshold on test is data leakage and
    #    inflates the reported score. The threshold is frozen here, then applied
    #    once to the held-out test year (2025).
    print("\n>>> Selecting threshold on validation set...")
    val_probs = trainer.predict(X_val, region_val)
    best_threshold, val_macro_f1, _ = find_optimal_threshold(y_val, val_probs)
    print(f"Threshold (chosen on val): {best_threshold:.2f}  |  Val Macro F1: {val_macro_f1:.3f}")

    # 8. Honest evaluation on the held-out test year at the frozen threshold.
    print("\n>>> Evaluating on held-out test set...")
    probs = trainer.predict(X_test, region_test)
    y_pred = (probs > best_threshold).astype(float)
    best_macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_macro_f1_at_05 = f1_score(y_test, (probs > 0.5).astype(float), average="macro")
    print(f"Test Macro F1 @{best_threshold:.2f}: {best_macro_f1:.3f}  |  @0.50: {test_macro_f1_at_05:.3f}")

    report_metrics = classification_report(y_test, y_pred, target_names=["Not Flyable", "Flyable"])

    # 9. Per-Region Evaluation on the held-out test set
    region_names = {i: f"Region_{i}" for i in range(config.num_regions)}
    region_results = evaluate_by_regions(model, test_loader, region_names)

    # 10. Save Experiment
    exp_dir = get_next_experiment_dir(config.experiments_dir)
    print(f"\n>>> Saving Experiment to: {exp_dir}")
    saver = ExperimentSaver(exp_dir)

    saver.save_model(model)
    saver.save_scaler(scaler)
    saver.save_features(feature_names)

    # Save predictions and artifacts
    test_results_df = test_df.copy()
    test_results_df["prob"] = probs
    test_results_df["pred"] = y_pred
    saver.save_predictions(test_results_df, best_threshold, config.hysteresis_margin)

    saver.save_region_mapping(region_mapping)

    cell_stats = evaluate_per_cell(test_results_df)
    saver.save_per_cell_stats(cell_stats)

    # Save Config and Summary
    summary = {
        "macro_f1": float(best_macro_f1),
        "macro_f1_at_0.5": float(test_macro_f1_at_05),
        "val_macro_f1": float(val_macro_f1),
        "threshold": float(best_threshold),
        "threshold_source": "validation (temporal)",
        "epochs_trained": trainer.epochs_trained,
        "best_val_loss": float(trainer.best_val_loss),
        "num_parameters": model.get_num_parameters(),
    }
    saver.save_config(config, summary)

    # Save plots
    plot_training_history(trainer.history, save_path=os.path.join(exp_dir, "training_history.png"))
    plot_threshold_analysis(
        y_test,
        probs,
        best_threshold,
        save_path=os.path.join(exp_dir, "threshold_analysis.png"),
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        ["Not Flyable", "Flyable"],
        title=f"CM (Th={best_threshold:.2f})",
        save_path=os.path.join(exp_dir, "confusion_matrix.png"),
    )

    plot_per_region_performance(
        region_results, save_path=os.path.join(exp_dir, "region_performance.png")
    )

    # Save text report
    report_text = f"""
Multi-Regional Model Training Report
{'='*50}
Date: {datetime.datetime.now()}
Experiment: {os.path.basename(exp_dir)}

Configuration:
  Number of regions: {config.num_regions}
  Regional embedding dim: {config.regional_embedding_dim}
  Learning rate: {config.learning_rate}
  Dropout rate: {config.dropout_rate}
  Confidence weighting: {config.use_confidence_weighting}

Results (honest, threshold chosen on validation):
  Test Macro F1 @threshold: {best_macro_f1:.3f}
  Test Macro F1 @0.50:      {test_macro_f1_at_05:.3f}
  Val  Macro F1 @threshold: {val_macro_f1:.3f}
  Threshold (from val):     {best_threshold:.2f}
  Epochs trained: {trainer.epochs_trained}
  Best validation loss: {trainer.best_val_loss:.4f}
  Model parameters: {model.get_num_parameters():,}

{report_metrics}
"""
    saver.save_report(report_text)

    print("\n✓ Experiment pipeline completed successfully.")
    return exp_dir


if __name__ == "__main__":
    # Can be run directly as a script if needed
    run_training_pipeline()
