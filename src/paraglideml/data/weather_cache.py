from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class WeatherCache:
    """
    Interface for accessing the GFS GRIB2 cache stored in NPZ format (V2).
    """

    def __init__(self, cache_root: str = "data/gfs/cache"):
        self.cache_root = Path(cache_root)

    def get_file_path(self, lat: int, lon: int, date: pd.Timestamp, hour: int) -> Path:
        cell_id = f"{lat}_{lon}"
        yyyy = date.strftime("%Y")
        mm = date.strftime("%m")
        yyyymmdd = date.strftime("%Y%m%d")
        hh = f"{hour:02d}"
        filename = f"gfsanl_3_{yyyymmdd}_{hh}00_000.npz"
        return self.cache_root / "cells" / cell_id / yyyy / mm / filename

    def load_sample(
        self, lat: int, lon: int, date: pd.Timestamp, hour: int
    ) -> Optional[Dict[str, Any]]:
        path = self.get_file_path(lat, lon, date, hour)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                keys = data["keys"]
                values = data["values"]
            raw_dict = {str(k): float(v) for k, v in zip(keys, values)}
            features = self._compute_derived_features(raw_dict)
            return {"raw": raw_dict, "features": features}
        except Exception:
            return None

    def _compute_derived_features(self, v: Dict[str, float]) -> Dict[str, float]:
        try:

            def get(key):
                return float(v.get(key, 0.0))

            # --- 1. Universal Profiles ---
            # We provide layer-by-layer data so the NN can "scan" the atmosphere.

            levels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500]
            profile_features = {}

            for i, p in enumerate(levels):
                # Wind Speed Profile (m/s)
                u = v.get(f"u_{p}hPa")
                v_wind = v.get(f"v_{p}hPa")  # renamed to avoid conflict with dict v
                if u is not None and v_wind is not None:
                    profile_features[f"ws_{p}"] = np.sqrt(u**2 + v_wind**2)
                else:
                    profile_features[f"ws_{p}"] = 0.0

                # Lapse Rate Profile (deg/km) - between current and next level
                if i < len(levels) - 1:
                    p_next = levels[i + 1]
                    t_bot, t_top = v.get(f"t_{p}hPa"), v.get(f"t_{p_next}hPa")
                    h_bot, h_top = v.get(f"gh_{p}hPa"), v.get(f"gh_{p_next}hPa")

                    if None not in [t_bot, t_top, h_bot, h_top]:
                        dz = h_top - h_bot
                        if dz > 50:
                            lr = (t_bot - t_top) / dz * 1000.0
                            profile_features[f"lr_{p}_{p_next}"] = np.clip(lr, -20, 40)
                        else:
                            profile_features[f"lr_{p}_{p_next}"] = 0.0
                    else:
                        profile_features[f"lr_{p}_{p_next}"] = 0.0

            # --- 2. Surface Context ---
            u10, v10 = get("10u_10m"), get("10v_10m")
            ws10 = np.sqrt(u10**2 + v10**2)

            # --- 3. Bulk / Legacy Features (Still valuable summaries) ---
            # Wind Shear Low (Surface vs 850hPa ~1.5km)
            u850, v850 = get("u_850hPa"), get("v_850hPa")
            ws850 = np.sqrt(u850**2 + v850**2)
            shear_low = np.sqrt((u850 - u10) ** 2 + (v850 - v10) ** 2)

            # 700hPa Wind
            u700, v700 = get("u_700hPa"), get("v_700hPa")
            ws700 = np.sqrt(u700**2 + v700**2)

            # Moisture (Dew Point Spread)
            dps850 = (100 - get("r_850hPa")) / 5.0
            dps700 = (100 - get("r_700hPa")) / 5.0

            features = {
                # Surface Anchor
                "surface_pressure": get("sp_0sfc"),
                "temp_2m": get("2t_2m"),
                # Surface Wind
                "u_10m": u10,
                "v_10m": v10,
                "wind_speed_10m": ws10,
                # Bulk Summaries
                "wind_speed_850": ws850,
                "wind_speed_700": ws700,
                "wind_shear_low": shear_low,  # Surf-850 shear
                "dps_850": dps850,
                "dps_700": dps700,
                "cape": get("cape_0sfc"),
                "cin": get("cin_0sfc"),
                "total_cloud_cover": get("tcc_0atm"),
            }

            # Merge profile
            features.update(profile_features)

            return features

        except Exception:
            return {}
