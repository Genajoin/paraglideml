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

            # --- 4. Convection / thermal proxies (previously cached but unused) ---
            # Vertical velocity (omega, Pa/s; negative = ascent, positive = subsidence).
            # A direct large-scale signal of convective forcing vs suppression.
            w850 = get("w_850hPa")
            w700 = get("w_700hPa")
            w600 = get("w_600hPa")
            w500 = get("w_500hPa")
            omega_low_mean = float(
                np.mean(
                    [
                        get("w_925hPa"),
                        get("w_900hPa"),
                        get("w_850hPa"),
                        get("w_800hPa"),
                        get("w_700hPa"),
                    ]
                )
            )

            # Surface gust (a primary flight-cancellation criterion) and visibility.
            gust_10m = get("gust_0sfc")
            visibility = get("vis_0sfc")

            # Surface dew-point spread (cloud-base / dryness proxy), in K.
            t2, td2 = get("2t_2m"), get("2d_2m")
            dewpoint_spread_2m = max(0.0, t2 - td2) if (t2 > 0 and td2 > 0) else 0.0

            # Low-level lapse-rate mean (boundary-layer instability proxy).
            low_lr_keys = ["lr_925_900", "lr_900_850", "lr_850_800", "lr_800_750", "lr_750_700"]
            low_lrs = [profile_features[k] for k in low_lr_keys if k in profile_features]
            lapse_low_mean = float(np.mean(low_lrs)) if low_lrs else 0.0

            # --- 5. Convective / thunderstorm-potential indices ---
            # Computed from the cached T/RH profile — NO new GFS fields needed. The model
            # already has CAPE/CIN/lapse (the instability ingredients); these add mid-level
            # moisture and storm-organisation signals that separate "good thermals" from
            # "overdevelops into thunderstorms" — the convective-false-positive failure mode
            # where flyable reads high on a day that actually storms out / is dangerous.
            def _tc(level):  # air temperature (deg C) at a pressure level, or None if missing
                t = v.get(f"t_{level}hPa")
                return (t - 273.15) if (t is not None and t > 150) else None

            def _td(level):  # dewpoint (deg C) from T and RH via Magnus, or None if missing
                t, rh = v.get(f"t_{level}hPa"), v.get(f"r_{level}hPa")
                if t is None or rh is None or t <= 150 or rh <= 0:
                    return None
                tc = t - 273.15
                rh = min(max(rh, 1.0), 100.0)
                g = np.log(rh / 100.0) + (17.625 * tc) / (243.04 + tc)
                return 243.04 * g / (17.625 - g)

            t850c, t700c, t500c = _tc(850), _tc(700), _tc(500)
            td850c, td700c = _td(850), _td(700)
            if None not in (t850c, t700c, t500c, td850c, td700c):
                vertical_totals = t850c - t500c          # mid-level lapse component
                cross_totals = td850c - t500c            # low-level moisture component
                total_totals = vertical_totals + cross_totals  # severe-storm index
                k_index = (t850c - t500c) + td850c - (t700c - td700c)  # thunderstorm/moisture
            else:
                vertical_totals = cross_totals = total_totals = k_index = 0.0

            # Mid-level RH (700-500 hPa): moist mid-levels fuel overdevelopment;
            # dry mid-levels give safe "blue" thermals.
            mid_rhs = [v.get(f"r_{p}hPa") for p in (700, 600, 500)]
            mid_rhs = [r for r in mid_rhs if r is not None and r > 0]
            mid_rh = float(np.mean(mid_rhs)) if mid_rhs else 0.0

            # Deep bulk shear surface->500 hPa (storm organisation / gust-front potential),
            # and CAPE x shear (an energy-shear severity proxy).
            u500, v500 = v.get("u_500hPa"), v.get("v_500hPa")
            deep_shear = (
                float(np.sqrt((u500 - u10) ** 2 + (v500 - v10) ** 2))
                if (u500 is not None and v500 is not None)
                else 0.0
            )
            cape_shear = get("cape_0sfc") * deep_shear

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
                # Convection / thermal proxies
                "w_850": w850,
                "w_700": w700,
                "w_600": w600,
                "w_500": w500,
                "omega_low_mean": omega_low_mean,
                "gust_10m": gust_10m,
                "visibility": visibility,
                "dewpoint_spread_2m": dewpoint_spread_2m,
                "lapse_low_mean": lapse_low_mean,
                # Convection / thunderstorm-potential indices (overdevelopment signal)
                "k_index": k_index,
                "total_totals": total_totals,
                "cross_totals": cross_totals,
                "mid_rh": mid_rh,
                "deep_shear": deep_shear,
                "cape_shear": cape_shear,
            }

            # Merge profile
            features.update(profile_features)

            return features

        except Exception:
            return {}
