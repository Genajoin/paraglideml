import datetime
import os

import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

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

    # 3. Create Data Loaders
    (
        train_loader,
        val_loader,
        scaler,
        X_train,
        y_train,
        region_train,
        conf_train,
        X_test,
        y_test,
        region_test,
        conf_test,
    ) = create_data_loaders(train_df, test_df, feature_names, config)

    # 4. Initialize Model
    model = MultiRegionalModel(
        input_dim=len(feature_names),
        num_regions=config.num_regions,
        embedding_dim=config.regional_embedding_dim,
        dropout_rate=config.dropout_rate,
    )

    # 5. Setup Training
    criterion = create_weighted_bce_loss(y_train, use_confidence=config.use_confidence_weighting)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    trainer = MultiRegionalTrainer(model, config, criterion, optimizer)

    # 6. Train
    print("\n>>> Starting Model Training...")
    trainer.fit(train_loader, val_loader, verbose=True)

    # 7. Evaluate and Find Optimal Threshold
    print("\n>>> Evaluating Model...")
    probs = trainer.predict(X_test, region_test)
    best_threshold, best_macro_f1, _ = find_optimal_threshold(y_test, probs)

    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Macro F1 Score: {best_macro_f1:.3f}")

    # Apply threshold
    y_pred = (probs > best_threshold).astype(float)
    report_metrics = classification_report(y_test, y_pred, target_names=["Not Flyable", "Flyable"])

    # 8. Per-Region Evaluation
    region_names = {i: f"Region_{i}" for i in range(config.num_regions)}
    region_results = evaluate_by_regions(model, val_loader, region_names)  # Use val or test loader

    # 9. Save Experiment
    exp_dir = get_next_experiment_dir(config.experiments_dir)
    print(f"\n>>> Saving Experiment to: {exp_dir}")
    saver = ExperimentSaver(exp_dir)

    saver.save_model(model)
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
        "threshold": float(best_threshold),
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

Results:
  Macro F1: {best_macro_f1:.3f}
  Optimal threshold: {best_threshold:.2f}
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
