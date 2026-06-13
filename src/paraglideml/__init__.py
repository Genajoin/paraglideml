"""
paraglideml — ML prediction of paragliding flight-quality from GFS weather.

Library inference API (torch-free; install `paraglideml[inference]`):

    from paraglideml import predict_tiers, tiers_to_geojson
    rows = predict_tiers("2026-06-12")              # per-cell P(>=flyable/good/epic)
    geojson = tiers_to_geojson(rows, "2026-06-12")  # 1-degree squares for the map

Defaults to the bundled exp_056 model; point a prod deployment at a fresher model with
$PARAGLIDEML_MODEL_DIR (see paraglideml.assets). Training / NN code lives behind the
`[train]` extra and is not imported here.
"""

from .predict import forecast_window, predict_tiers, tiers_to_geojson
from .tiers import TIER_LABELS, TIERS

__all__ = ["predict_tiers", "forecast_window", "tiers_to_geojson", "TIERS", "TIER_LABELS"]
