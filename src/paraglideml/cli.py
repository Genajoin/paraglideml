import json
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

# NOTE: heavy training/NN imports (.train -> torch) are done lazily inside the commands
# that need them, so the `paraglideml` CLI and the lean inference path start without torch.

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


@data_app.command("backfill")
def data_backfill(
    start: str = typer.Option(..., "--start", help="Window start YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Window end YYYY-MM-DD (inclusive)"),
    months: str = typer.Option("3,4,5,6,7,8,9,10", help="Months to fill, comma-separated"),
    extract_cells: Path = typer.Option(
        PROCESSED_DATA_DIR / "extract_cells.json", help="JSON list of cells to extract"
    ),
    cache_dir: Path = typer.Option(GFS_CACHE_DIR, help="Weather cache root to fill"),
    archive_dir: Optional[Path] = typer.Option(
        None, "--archive-dir", help="Keep raw GRIB here (~105 MB/slice) instead of deleting"
    ),
    workers: int = typer.Option(3, help="Slices downloaded/extracted in parallel"),
):
    """
    Backfill the GFS analysis cache over a date window (fetch -> extract -> keep/drop GRIB).

    Traffic depends only on the number of (date, hour) slices — ~105 MB each, three
    per day — not on the cell count, since each slice carries global GRIB fields.
    Resumable: slices already satisfied are skipped.

    With --archive-dir the raw GRIB is kept (put it on a spinning disk: a full
    March-October 2021-2026 window is ~434 GB). That is what makes a later region
    expansion free — new cells re-extract off the archive via `data gfs`, with no
    download. Without it, only the cells chosen today survive, and widening the
    region later means pulling all ~434 GB again.
    """
    from .data.backfill import run_backfill

    cells = json.loads(Path(extract_cells).read_text()) if Path(extract_cells).exists() else None
    run_backfill(
        start=start,
        end=end,
        cells=cells,
        months=tuple(int(m) for m in months.split(",")),
        cache_root=cache_dir,
        archive_root=archive_dir,
        workers=workers,
    )


@data_app.command("terrain")
def data_terrain(
    selected_cells: Path = typer.Option(
        PROCESSED_DATA_DIR / "selected_cells.json", help="JSON file with selected cells"
    ),
    sites_dir: Optional[str] = typer.Option(
        None, help="FlyBeeper spot json dir (default: config.FLYBEEPER_SITES_DIR)"
    ),
):
    """
    Aggregate FlyBeeper launch sites into per-cell terrain & slope orientations.

    Uses known site altitudes (dhv_loc.geojson) and launch-orientation flags
    (takeoff.geojson) to write data/processed/cell_terrain.json, which the dataset
    builder merges as features (elevation, mountainess, slope-wind alignment).
    """
    from .data.terrain import build_cell_terrain

    typer.echo("Aggregating FlyBeeper spot terrain & orientations...")
    build_cell_terrain(selected_cells_path=str(selected_cells), sites_dir=sites_dir)


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
    from .train import run_training_pipeline  # lazy: pulls torch only when training

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


@train_app.command("goodxc")
def train_goodxc(
    experiments_dir: str = typer.Option(
        "models/experiments", help="Directory to save experiment artifacts"
    ),
    good_km: float = typer.Option(50.0, help="Distance (km) that marks a 'good XC day'"),
    broad_min: int = typer.Option(
        5, help="Cells in a region reaching good_km for the day to count as broadly good"
    ),
    drop_middle: bool = typer.Option(
        False, "--drop-middle", help="Hard-exclude the ambiguous middle from training"
    ),
    target_precision: float = typer.Option(
        0.80, help="Precision target for the anti-noise operating point (chosen on val)"
    ),
):
    """
    Train the distance-based P(good XC day) model — the product target for the bot.

    Reports Average Precision / calibration on the held-out year, alongside the AP
    of the old is_flyable label on the same features, to show whether distance is a
    cleaner, more weather-predictable target.
    """
    from .goodxc import run_goodxc_pipeline

    print("Starting P(good XC day) pipeline...")
    try:
        exp_path = run_goodxc_pipeline(
            experiments_dir=experiments_dir,
            good_km=good_km,
            broad_min=broad_min,
            drop_middle=drop_middle,
            target_precision=target_precision,
        )
        print(f"\ngood-xc pipeline finished. Experiment saved to: {exp_path}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Please ensure the dataset exists at the configured path.", err=True)
    except Exception as e:
        typer.echo(f"An error occurred during good-xc training: {e}", err=True)


@train_app.command("backtest")
def train_backtest(
    broad_min: int = typer.Option(5, help="Regional-consensus broad threshold"),
):
    """
    Rolling-origin backtest of the good-XC recipe: for each year, fit on all prior
    years and score on that year. Reports AP/ROC per tier per year + the mean — an
    estimate of how the recipe generalizes to any unseen season.
    """
    from .goodxc import run_backtest

    print("Starting rolling backtest...")
    try:
        run_backtest(broad_min=broad_min)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
    except Exception as e:
        typer.echo(f"An error occurred during backtest: {e}", err=True)


@train_app.command("ordinal")
def train_ordinal(
    experiments_dir: str = typer.Option(
        "models/experiments", help="Directory to save experiment artifacts"
    ),
    broad_min: int = typer.Option(
        5, help="Cells in a region reaching a tier for the day to count as broadly good"
    ),
    production: bool = typer.Option(
        False, "--production", help="Fit on ALL years (no holdout) for deployment; "
        "quality is estimated via `train backtest`, not a held-out test"
    ),
):
    """
    Train calibrated ordinal flight-quality tiers: cumulative P(>=flyable/good/epic).

    Three distance thresholds (15/50/100 km) trained as calibrated binary models with
    regional-consensus confidence, reported with AP/ROC/Brier and a monotonicity check.
    """
    from .ordinal import run_ordinal_pipeline

    print("Starting ordinal tier pipeline...")
    try:
        exp_path = run_ordinal_pipeline(
            experiments_dir=experiments_dir, broad_min=broad_min, production=production
        )
        print(f"\nOrdinal pipeline finished. Experiment saved to: {exp_path}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Please ensure the dataset exists at the configured path.", err=True)
    except Exception as e:
        typer.echo(f"An error occurred during ordinal training: {e}", err=True)


@app.command("forecast")
def forecast(
    date: str = typer.Option(..., "--date", help="Target date YYYY-MM-DD"),
    experiment: Optional[str] = typer.Option(
        None, "--experiment", help="Experiment name to load (default: latest NN with model.pth)"
    ),
    grib_dir: Optional[str] = typer.Option(
        None,
        "--grib-dir",
        help="Where to download GRIB (default: NVMe data/gfs/forecast_grib; "
        "point at a big disk for large backfills)",
    ),
):
    """
    Download GFS for a date, run the trained model, and print per-spot flyability.

    Fetches only the needed GRIB messages (byte-range). v1 uses the 0.25 deg
    analysis (real conditions for the date) — ideal for eyeballing the model
    against a recent day.
    """
    from .predict import run_forecast

    try:
        run_forecast(date_str=date, experiment=experiment, grib_dir=grib_dir)
    except Exception as e:
        typer.echo(f"Forecast failed: {e}", err=True)


@app.command("forecast-tiers")
def forecast_tiers(
    date: str = typer.Option(..., "--date", help="Target date YYYY-MM-DD"),
    experiment: Optional[str] = typer.Option(
        None, "--experiment", help="Dev: use models/experiments/<name> instead of the bundled model"
    ),
    model_dir: Optional[str] = typer.Option(
        None, "--model-dir", help="Model dir override (else $PARAGLIDEML_MODEL_DIR or bundled exp_056)"
    ),
    grib_dir: Optional[str] = typer.Option(None, "--grib-dir", help="Where to download GRIB"),
    push_threshold: float = typer.Option(
        0.5, help="P(>=good) threshold for the bot's push decision in the table"
    ),
    geojson: Optional[str] = typer.Option(
        None, "--geojson", help="Also write the artifact (1-degree squares + tiers) to this path"
    ),
):
    """
    Forecast ordinal flight-quality tiers for a date: per-spot P(>=flyable/good/epic).

    Cumulative calibrated probabilities from the bundled exp_056 model (or an override).
    The bot pushes on P(>=good); --geojson emits the map/pipeline artifact for the date.
    """
    from .predict import run_ordinal_forecast

    try:
        run_ordinal_forecast(
            date_str=date, experiment=experiment, model_dir=model_dir, grib_dir=grib_dir,
            push_threshold=push_threshold, geojson_out=geojson,
        )
    except Exception as e:
        typer.echo(f"Tier forecast failed: {e}", err=True)


@app.command("forecast-window")
def forecast_window_cmd(
    run_date: str = typer.Option(..., "--run-date", help="GFS run date YYYY-MM-DD (the 00z cycle to forecast from)"),
    days: int = typer.Option(3, "--days", help="Horizon in days: run_date+1 .. run_date+days"),
    out: str = typer.Option(..., "--out", help="Write the multi-day GeoJSON artifact to this path"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Model dir override (else bundled exp_056)"),
    grib_dir: Optional[str] = typer.Option(None, "--grib-dir", help="Where to download forecast GRIB"),
):
    """
    Produce the production artifact: per-cell P(>=flyable/good/epic) for the next `days`
    days, each scored from its forecast lead-time off the run_date 00z cycle. Writes a
    GeoJSON FeatureCollection of 1-degree squares (cells x days) — the R2 / map contract.
    """
    import json as _json

    from .predict import forecast_window, tiers_to_geojson

    try:
        rows = forecast_window(run_date, days=days, model_dir=model_dir, grib_dir=grib_dir)
        gj = tiers_to_geojson(rows)
        Path(out).write_text(_json.dumps(gj))
        typer.echo(f"Wrote {len(gj['features'])} features ({days} days x cells) to {out}")
    except Exception as e:
        typer.echo(f"Forecast-window failed: {e}", err=True)
        raise


@app.command("forecast-skew")
def forecast_skew(
    start: str = typer.Option(..., "--start", help="Window start YYYY-MM-DD (valid days with known outcomes)"),
    end: str = typer.Option(..., "--end", help="Window end YYYY-MM-DD"),
    leads: str = typer.Option("1,3,5", "--leads", help="Forecast lead-times in DAYS, comma-separated"),
    sample_every: int = typer.Option(1, "--sample-every", help="Use every Nth day in the window (cut downloads)"),
    experiment: Optional[str] = typer.Option(None, "--experiment", help="Ordinal experiment (default: latest production)"),
    scratch_root: Optional[str] = typer.Option(None, "--scratch", help="Where to stash forecast GRIB+cache"),
):
    """
    Measure how much AP/ROC drops when the ordinal model is fed GFS *forecast*
    lead-times (the bot's reality) instead of the *analysis* it trained on.

    Downloads, for each lead L, the same valid days from the run issued L days earlier
    and re-scores on the identical cell-days. Decides v1 deploy: ship-as-is on short
    horizons vs forecast-aware retrain. WARNING: ~107 MB per slice, 3 slices/day/lead.
    """
    from .skew import run_forecast_skew

    lead_list = [int(x) for x in leads.split(",") if x.strip()]
    try:
        run_forecast_skew(
            start_date=start, end_date=end, leads=lead_list,
            experiment=experiment, sample_every=sample_every, scratch_root=scratch_root,
        )
    except Exception as e:
        typer.echo(f"Forecast-skew failed: {e}", err=True)
        raise


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
