import glob
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Regex to extract coordinates from XContest takeoff link
# Example: ...filter[point]=13.779207 46.181068...
COORD_REGEX = re.compile(r"filter\[point\]=([\d\.]+)\s+([\d\.]+)")


def parse_flight(flight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single flight dictionary into a flat structure.
    """
    flat_flight = {
        "id": flight.get("id"),
        "pilot_id": flight.get("pilot", {}).get("id"),
        "pilot_name": flight.get("pilot", {}).get("name"),
        "date": flight.get("pointStart", {}).get("time"),
        "distance": flight.get("league", {}).get("route", {}).get("distance"),
        "points": flight.get("league", {}).get("route", {}).get("points"),
        "glider": flight.get("glider", {}).get("name"),
        "glider_class": flight.get("glider", {}).get("class"),
        "takeoff_name": flight.get("takeoff", {}).get("name"),
        "country": flight.get("takeoff", {}).get("countryIso"),
    }

    # Extract coordinates
    takeoff_link = flight.get("takeoff", {}).get("link", "")
    if takeoff_link:
        match = COORD_REGEX.search(takeoff_link)
        if match:
            # XContest usually provides LON LAT
            flat_flight["takeoff_lon"] = float(match.group(1))
            flat_flight["takeoff_lat"] = float(match.group(2))
        else:
            flat_flight["takeoff_lon"] = None
            flat_flight["takeoff_lat"] = None
    else:
        flat_flight["takeoff_lon"] = None
        flat_flight["takeoff_lat"] = None

    return flat_flight


def load_flights_to_dataframe(
    data_dir: str = "data/flights", pattern: str = "*.json"
) -> pd.DataFrame:
    """
    Loads all JSON flight files from the directory matching the pattern into a Pandas DataFrame.
    """
    files = glob.glob(str(Path(data_dir) / pattern))
    all_flights = []

    print(f"Found {len(files)} files in {data_dir} matching {pattern}")

    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Data is expected to be a list of flights
                if isinstance(data, list):
                    for flight in data:
                        all_flights.append(parse_flight(flight))
                else:
                    print(f"Warning: File {file_path} does not contain a list.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    df = pd.DataFrame(all_flights)

    # Post-processing
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

    return df


if __name__ == "__main__":
    # Test run
    df = load_flights_to_dataframe()
    print(df.head())
    print(df.info())
