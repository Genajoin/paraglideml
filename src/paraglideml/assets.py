"""
Asset resolution: where inference loads the model and supporting data from.

Default = the assets bundled inside the package (exp_056), so `pip install paraglideml`
yields a working predictor out of the box (demo + first deploy). Override at runtime via
env to point a prod deployment at a fresher model (on disk / synced from R2) WITHOUT
re-releasing the library — code cadence (rare) is decoupled from model cadence (every
retrain). Weights live as data, versioned and deployed independently of the package.

  PARAGLIDEML_MODEL_DIR      -> dir with model_{flyable,good,epic}.joblib + calibrator_* + features.txt
  PARAGLIDEML_CELL_TERRAIN   -> cell_terrain.json (spot-centric elevation / mountainess)
  PARAGLIDEML_SELECTED_CELLS -> selected_cells.json (the cells to score)
"""

import os
from importlib import resources
from pathlib import Path


def _bundled(*parts: str) -> Path:
    """Path to a file under the package's bundled assets/ dir (works from a wheel)."""
    p = resources.files("paraglideml") / "assets"
    for part in parts:
        p = p / part
    return Path(str(p))


def model_dir() -> Path:
    """Ordinal model dir: $PARAGLIDEML_MODEL_DIR, else the bundled exp_056."""
    env = os.getenv("PARAGLIDEML_MODEL_DIR")
    return Path(env) if env else _bundled("model")


def terrain_path() -> Path:
    """cell_terrain.json: $PARAGLIDEML_CELL_TERRAIN, else bundled."""
    env = os.getenv("PARAGLIDEML_CELL_TERRAIN")
    return Path(env) if env else _bundled("cell_terrain.json")


def cells_path() -> Path:
    """selected_cells.json: $PARAGLIDEML_SELECTED_CELLS, else bundled."""
    env = os.getenv("PARAGLIDEML_SELECTED_CELLS")
    return Path(env) if env else _bundled("selected_cells.json")
