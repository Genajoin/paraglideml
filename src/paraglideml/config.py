import os
from pathlib import Path

# Base Project Paths
PACKAGE_DIR = Path(__file__).parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def load_env(env_path: Path):
    """Simple .env loader that doesn't override existing env vars"""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


# Load .env from project root
load_env(PROJECT_ROOT / ".env")

# --- PATH CONFIGURATION ---
DATA_DIR = Path(os.getenv("PARAGLIDEML_DATA_DIR", PROJECT_ROOT / "data"))

# Sub-directories (can be overridden individually)
GFS_DIR = Path(os.getenv("PARAGLIDEML_GFS_DIR", DATA_DIR / "gfs"))
GFS_ANL_DIR = Path(os.getenv("PARAGLIDEML_GFS_ANL_DIR", GFS_DIR / "anl"))
GFS_CACHE_DIR = Path(os.getenv("PARAGLIDEML_GFS_CACHE_DIR", GFS_DIR / "cache"))
# Landing dir for on-demand forecast/inference GRIB downloads. Small samples stay
# on fast local storage (NVMe); point this at a large/slow disk (HDD) for big
# history backfills via PARAGLIDEML_FORECAST_GRIB_DIR or the --grib-dir flag.
GFS_FORECAST_DIR = Path(os.getenv("PARAGLIDEML_FORECAST_GRIB_DIR", GFS_DIR / "forecast_grib"))
PROCESSED_DATA_DIR = Path(os.getenv("PARAGLIDEML_PROCESSED_DATA_DIR", DATA_DIR / "processed"))
FLIGHTS_DIR = Path(os.getenv("PARAGLIDEML_FLIGHTS_DIR", DATA_DIR / "flights"))

# Elevation GeoTIFF for spot-centric terrain features (launch-point elevation /
# mountainess). Default is the global ETOPO 2022 raster from the Paraglidable
# archive (~1.8 km, full coverage); override with a finer SRTM mosaic if available.
ELEVATION_TIF = Path(
    os.getenv(
        "PARAGLIDEML_ELEVATION_TIF",
        "/home/gena/archive/dev/Paraglidable/data/elevation/ETOPO_2022_v1_60s_N90W180_bed.tif",
    )
)
# Per-cell terrain table produced by `paraglideml data terrain` (small JSON; the
# training/inference path reads this, never the heavy raster).
CELL_TERRAIN_PATH = Path(
    os.getenv("PARAGLIDEML_CELL_TERRAIN", str(PROCESSED_DATA_DIR / "cell_terrain.json"))
)

MODELS_DIR = Path(os.getenv("PARAGLIDEML_MODELS_DIR", PROJECT_ROOT / "models"))
EXPERIMENTS_DIR = Path(os.getenv("PARAGLIDEML_EXPERIMENTS_DIR", MODELS_DIR / "experiments"))

# --- DATA PROCESSING CONFIGURATION ---
DEFAULT_DATES = os.getenv("PARAGLIDEML_DATES", "2024-01-01:2024-12-31")
# Format: "lon_min,lat_min,lon_max,lat_max" (e.g. Slovenia/Alps region)
DEFAULT_BBOX = os.getenv("PARAGLIDEML_DEFAULT_BBOX", "13.0,45.0,15.0,47.0")
# Minimum XContest points for a flight to be considered "quality XC"
DEFAULT_MIN_XC_POINTS = int(os.getenv("PARAGLIDEML_MIN_XC_POINTS", "10"))

# --- TRAINING DEFAULTS ---
DEFAULT_NUM_REGIONS = int(os.getenv("PARAGLIDEML_NUM_REGIONS", "3"))
DEFAULT_EPOCHS = int(os.getenv("PARAGLIDEML_EPOCHS", "150"))
DEFAULT_BATCH_SIZE = int(os.getenv("PARAGLIDEML_BATCH_SIZE", "32"))
DEFAULT_LEARNING_RATE = float(os.getenv("PARAGLIDEML_LEARNING_RATE", "0.001"))
