# Paraglideml

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paraglideml** is an ML system for forecasting paragliding flight conditions based on GFS (Global Forecast System) meteorological data and historical flight tracks from XContest.

## Project

The goal is to predict flyability for specific locations: **Flyable / Not Flyable**.

Current focus: the Alpine region (Slovenia, Italy, Austria). The model is architected to generalise to any mountain or flatland sites.

![Alps Region](docs/alps.png)

---

## Quick Start (Example Data)

To quickly verify a working setup on the bundled example data for the Kobala launch (Slovenia):

```bash
pip install -e .
cp .env.example .env
paraglideml train model
```

## Data Acquisition

For full operation (beyond the example mode), the following data needs to be prepared:

1. **Weather (GFS)**: download GFS Analysis archives (0.25 degree) in `.grb2` format.
   - Source: [NOAA GFS S3](https://noaa-gfs-bdp-pds.s3.amazonaws.com/)
   - Project path: `data/gfs/anl/YYYY-MM/` (configurable via `.env`)
   - Files: `gfsanl_3_YYYYMMDD_HH00_000.grb2`

2. **Flights (XContest)**: export flight data for your period and area of interest in `.json` format.
   - Source: [XContest](https://www.xcontest.org/)
   - Project path: `data/flights/` (configurable via `.env`)

You can use the downloaders from the [PyParaglide](https://github.com/Genajoin/PyParaglide) project.

## Full Pipeline

The project uses a **CLI-first** approach. All main operations are run via the `paraglideml` command.

```bash
# Install
pip install -e .

# Check configuration
paraglideml info

# Pipeline steps:
# 1. Prepare GFS data
paraglideml data gfs

# 2. Analyse flights and select quality cells
paraglideml data flights

# 3. Build training dataset
paraglideml data build

# 4. Train the model
paraglideml train model
```

---

## Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                             │
├─────────────────────────────────────────────────────────────────┤
│ • GFS caching: paraglideml data gfs                             │
│   → Extracts 135+ parameters from GRIB2 into NPZ                │
│   → Uses date and region settings from .env                     │
│                                                                 │
│ • Cell analysis: paraglideml data flights                       │
│   → Computes cell quality (flights, coverage)                   │
│   → Creates data/processed/selected_cells.json                  │
│                                                                 │
│ • Dataset build: paraglideml data build                         │
│   → Joins weather + flights into multicell_dataset.csv          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. MODEL TRAINING                                               │
├─────────────────────────────────────────────────────────────────┤
│ • MultiRegional Model: paraglideml train model                  │
│   → Architecture: Regional Attention + Confidence Weighting     │
│   → Optimises Macro F1 Score                                    │
│   → Output: models/experiments/exp_XXX/                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. RESULTS ANALYSIS                                             │
├─────────────────────────────────────────────────────────────────┤
│ • Summary:  paraglideml analyze summary [exp_XXX]               │
│             (defaults to the latest experiment)                 │
│ • Errors:   paraglideml analyze errors [exp_XXX]                │
│             (defaults to the latest experiment)                 │
│ • Compare:  paraglideml analyze compare --limit 5               │
│ • Notebook: notebooks/05_multiregional_model.ipynb              │
│                                                                 │
│ Artifacts (in experiment folder):                               │
│   ├── model.pth, config.json, report.txt                        │
│   ├── training_history.png, confusion_matrix.png                │
│   ├── tp.csv, tn.csv, fp.csv, fn.csv                            │
│   └── per_cell_stats.csv                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model: Multi-Regional Attention

The current model version (`src/paraglideml/multiregional.py`) addresses the problem of geographic variability in flight conditions.

**Key features:**
- **K-means clustering** — groups cells into regions based on coordinates.
- **Regional Embedding** — learnable vectors encoding region-specific characteristics.
- **Multi-head Attention** — attention mechanism adapting general weather features to a specific region.
- **Confidence Weighting** — weighting training examples by label confidence (flyable / non-flyable day).

More details: `docs/multiregional.md`

---

## Project Structure

```
paraglideml/
├── .env                          # Path and parameter configuration
├── pyproject.toml
├── README.md
├── MODEL.md                      # Feature documentation
├── CLAUDE.md / GEMINI.md         # AI agent instructions
│
├── src/paraglideml/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point (typer)
│   ├── config.py                 # Configuration management
│   ├── multiregional.py          # Model architecture and utilities
│   ├── train.py                  # Training pipeline
│   │
│   ├── data/                     # Data processing
│   │   ├── gfs_processor.py      # GRIB2 → NPZ processing
│   │   ├── cell_analyzer.py      # Cell analysis
│   │   ├── dataset_builder.py    # Dataset assembly
│   │   ├── flight_parsing.py     # XContest parsing
│   │   └── weather_cache.py      # NPZ cache reader
│   │
│   └── analysis/                 # Analysis tools
│       ├── summary.py            # Reports and comparison
│       └── error_analyzer.py     # Detailed error analysis
│
├── notebooks/                    # Jupyter notebooks
│   └── 05_multiregional_model.ipynb # Visual analysis and experiments
│
├── models/                       # Training results
│   └── experiments/              # Experiments (exp_XXX/)
│
├── scripts/                      # Helper scripts
│   └── archive/                  # Deprecated versions
│
└── data/                         # Data
    ├── gfs/anl/                  # Raw GRIB2 files
    ├── gfs/cache/                # Processed NPZ cache
    ├── flights/                  # Flight logs (JSON)
    └── processed/                # CSV datasets and metadata
```

---

## Configuration

All settings (paths, training parameters, date ranges) are managed via the `.env` file in the project root.
To inspect the current configuration:

```bash
paraglideml info
```

---

## Development

```bash
# Code formatting
black src/
isort src/

# Editable install
pip install -e .
```

---

## License

MIT

---

## Hire me

I take on commercial engineering work through **[Alpisto d.o.o.](https://alpisto.eu)** (Slovenia, EU) — MATLAB → Python migrations, power-systems algorithms, embedded BLE/RTOS firmware, and IoT backends.

→ [alpisto.eu/matlab-to-python](https://alpisto.eu/matlab-to-python) · **gena@alpisto.eu** · [LinkedIn](https://www.linkedin.com/in/evgenyistomin/)

