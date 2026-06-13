from pathlib import Path
from typing import Optional

import typer

from .analysis.error_analyzer import analyze_experiment
from .analysis.summary import analyze_experiment_results, compare_experiments
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BBOX,
    DEFAULT_DATES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_XC_POINTS,
    DEFAULT_NUM_REGIONS,
    FLIGHTS_DIR,
    GFS_ANL_DIR,
    GFS_CACHE_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from .data.cell_analyzer import get_cell_statistics, select_quality_cells
from .data.dataset_builder import build_multicell_dataset
from .data.gfs_processor import run_gfs_cache_creation
from .train import run_training_pipeline

app = typer.Typer(help="Paraglideml CLI: Machine Learning for Paragliding")
data_app = typer.Typer(help="Data processing and cache management")
train_app = typer.Typer(help="Model training commands")
analyze_app = typer.Typer(help="Analysis and evaluation tools")

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(analyze_app, name="analyze")


@app.command()
def info():
    """
    Display project information and current configuration.
    """
    from .config import (
        DATA_DIR,
        DEFAULT_BATCH_SIZE,
        DEFAULT_BBOX,
        DEFAULT_DATES,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MIN_XC_POINTS,
        DEFAULT_NUM_REGIONS,
        EXPERIMENTS_DIR,
        FLIGHTS_DIR,
        GFS_ANL_DIR,
        GFS_CACHE_DIR,
        PROCESSED_DATA_DIR,
    )

    typer.echo("=== Paraglideml Project Info ===")
    typer.echo("Version: 0.1.0")
    typer.echo("\n[Paths]")
    typer.echo(f"  Data Root:      {DATA_DIR}")
    typer.echo(f"  GFS Raw:        {GFS_ANL_DIR}")
    typer.echo(f"  GFS Cache:      {GFS_CACHE_DIR}")
    typer.echo(f"  Processed Data: {PROCESSED_DATA_DIR}")
    typer.echo(f"  Flights:        {FLIGHTS_DIR}")
    typer.echo(f"  Models Root:    {MODELS_DIR}")
    typer.echo(f"  Experiments:    {EXPERIMENTS_DIR}")

    typer.echo("\n[Defaults]")
    typer.echo(f"  Date Ranges:    {DEFAULT_DATES}")
    typer.echo(f"  BBox:           {DEFAULT_BBOX}")
    typer.echo(f"  Min XC Points:  {DEFAULT_MIN_XC_POINTS}")
    typer.echo(f"  Regions:        {DEFAULT_NUM_REGIONS}")
    typer.echo(f"  Epochs:         {DEFAULT_EPOCHS}")
    typer.echo(f"  Batch Size:     {DEFAULT_BATCH_SIZE}")
    typer.echo(f"  Learning Rate:  {DEFAULT_LEARNING_RATE}")


# =============================================================================
# DATA COMMANDS
# =============================================================================


@data_app.command("gfs")
def data_cache_gfs(
    dates: str = typer.Option(
        DEFAULT_DATES,
        help="Date ranges (e.g., '2024-01-01:2024-01-31,2024-03-01:2024-03-15')",
    ),
    bbox: str = typer.Option(DEFAULT_BBOX, help="Bounding box 'lon_min,lat_min,lon_max,lat_max'"),
    source_dir: Path = typer.Option(GFS_ANL_DIR, help="Source GFS directory"),
    output_dir: Path = typer.Option(GFS_CACHE_DIR, help="Output cache directory"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing cache"),
):
    """
    Process GFS GRIB2 files into local NPZ cache for specific cells.
    """
    typer.echo("Starting GFS cache creation...")
    run_gfs_cache_creation(dates, bbox, source_dir, output_dir, force)


@data_app.command("flights")
def data_prepare(
    flights_dir: Path = typer.Option(FLIGHTS_DIR, help="Directory with flight logs"),
    cache_dir: Path = typer.Option(GFS_CACHE_DIR, help="Directory with GFS cache"),
    output_csv: Path = typer.Option(
        PROCESSED_DATA_DIR / "cell_quality.csv", help="Output CSV with cell stats"
    ),
    output_json: Path = typer.Option(
        PROCESSED_DATA_DIR / "selected_cells.json",
        help="Output JSON with selected cells",
    ),
    min_flights: int = typer.Option(200, help="Minimum total flights"),
    min_flyable_days: int = typer.Option(30, help="Minimum flyable days"),
    min_coverage: float = typer.Option(80.0, help="Minimum weather coverage percent"),
    bbox: str = typer.Option(DEFAULT_BBOX, help="Bounding box 'lon_min,lat_min,lon_max,lat_max'"),
    no_bbox_filter: bool = typer.Option(
        False, "--no-bbox-filter", help="Disable bbox filtering (analyze all cells)"
    ),
    min_xc_points: int = typer.Option(
        DEFAULT_MIN_XC_POINTS, help="Minimum XContest points for quality XC flight"
    ),
):
    """
    Analyze flight logs and weather coverage to select quality cells for training.
    """
    import json

    typer.echo("Analyzing cells...")
    get_cell_statistics(
        flights_dir=str(flights_dir),
        cache_root=str(cache_dir),
        output_path=str(output_csv),
        bbox=bbox if not no_bbox_filter else None,
        min_xc_points=min_xc_points,
    )

    typer.echo("\nSelecting best cells...")
    selected = select_quality_cells(
        cell_quality_path=str(output_csv),
        min_flights=min_flights,
        min_flyable_days=min_flyable_days,
        min_weather_coverage=min_coverage,
        min_regions=DEFAULT_NUM_REGIONS,
    )

    with open(output_json, "w") as f:
        json.dump(selected, f, indent=2)
    typer.echo(f"\nSaved {len(selected)} selected cells to {output_json}")


@data_app.command("build")
def data_build(
    selected_cells: Path = typer.Option(
        PROCESSED_DATA_DIR / "selected_cells.json", help="JSON file with selected cells"
    ),
    flights_dir: Path = typer.Option(FLIGHTS_DIR, help="Directory with flight logs"),
    cache_dir: Path = typer.Option(GFS_CACHE_DIR, help="Directory with GFS cache"),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "multicell_dataset.csv", help="Output CSV dataset path"
    ),
    min_xc_points: int = typer.Option(
        DEFAULT_MIN_XC_POINTS, help="Minimum XContest points for quality XC flight"
    ),
):
    """
    Build the final multi-cell dataset from selected cells and weather cache.
    """
    typer.echo("Building dataset...")
    build_multicell_dataset(
        selected_cells_path=str(selected_cells),
        flights_dir=str(flights_dir),
        cache_root=str(cache_dir),
        output_path=str(output_path),
        min_xc_points=min_xc_points,
    )


# =============================================================================
# TRAIN COMMANDS
# =============================================================================


@train_app.command("model")
def train_model(
    num_regions: int = typer.Option(DEFAULT_NUM_REGIONS, help="Number of regions for clustering"),
    epochs: int = typer.Option(DEFAULT_EPOCHS, help="Number of training epochs"),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE, help="Learning rate for the optimizer"
    ),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Batch size for training"),
    experiments_dir: str = typer.Option(
        "models/experiments", help="Directory to save experiment artifacts"
    ),
):
    """
    Train the multi-regional model using the defined pipeline.
    """
    print("Starting training pipeline with:")
    print(f"  Regions: {num_regions}, Epochs: {epochs}, LR: {learning_rate}, BS: {batch_size}")

    try:
        exp_path = run_training_pipeline(
            num_regions=num_regions,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            experiments_dir=experiments_dir,
        )
        print(f"\nTraining pipeline finished. Experiment saved to: {exp_path}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Please ensure the dataset exists at the configured path.", err=True)
    except Exception as e:
        typer.echo(f"An error occurred during training: {e}", err=True)


@train_app.command("baseline")
def train_baseline(
    experiments_dir: str = typer.Option(
        "models/experiments", help="Directory to save experiment artifacts"
    ),
    learning_rate: float = typer.Option(0.05, help="Gradient boosting learning rate"),
    max_iter: int = typer.Option(400, help="Max boosting iterations (trees)"),
    max_leaf_nodes: int = typer.Option(31, help="Max leaf nodes per tree (capacity)"),
):
    """
    Train a gradient-boosted baseline (HistGradientBoostingClassifier) on the same
    features and honest protocol as the NN, to establish the realistic ceiling.
    """
    from .baseline import run_baseline_pipeline

    print("Starting gradient-boosted baseline...")
    try:
        exp_path = run_baseline_pipeline(
            experiments_dir=experiments_dir,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
        )
        print(f"\nBaseline finished. Experiment saved to: {exp_path}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Please ensure the dataset exists at the configured path.", err=True)
    except Exception as e:
        typer.echo(f"An error occurred during baseline training: {e}", err=True)


# =============================================================================
# ANALYZE COMMANDS
# =============================================================================


@analyze_app.command("errors")
def analyze_errors(
    exp_name: Optional[str] = typer.Argument(
        None, help="Experiment folder name (e.g., exp_001). Defaults to latest."
    ),
):
    """
    Detailed analysis of model errors (influential features, neutral zone).
    """
    analyze_experiment(exp_name)


@analyze_app.command("summary")
def analyze_summary(
    exp_name: Optional[str] = typer.Argument(
        None, help="Experiment folder name (e.g., exp_001). Defaults to latest."
    ),
):
    """
    Print a summary of experiment results and per-cell performance.
    """
    analyze_experiment_results(exp_name)


@analyze_app.command("compare")
def analyze_compare(
    limit: int = typer.Option(5, help="Number of recent experiments to compare"),
):
    """
    Compare multiple experiments against each other and the baseline.
    """
    compare_experiments(limit=limit)


if __name__ == "__main__":
    app()
