from __future__ import annotations

import os
import time
import requests
import pandas as pd

# Rough ERCOT coverage via major TX metro coordinates
# (lat, lon, label)
POINTS = [
    (29.7604, -95.3698, "houston"),
    (32.7767, -96.7970, "dallas"),
    (30.2672, -97.7431, "austin"),
    (29.4241, -98.4936, "san_antonio"),
    (27.8006, -97.3964, "corpus_christi"),
    (31.7619, -106.4850, "el_paso"),
    (25.9017, -97.4975, "brownsville"),
    (33.5779, -101.8552, "lubbock"),
]

START = "2018-01-01"
END = "2024-12-31"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_point(lat: float, lon: float, label: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Chicago",  # close enough for ERCOT; we’ll keep timestamps consistent
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return pd.DataFrame(columns=["timestamp", "temp_f", "point"])

    df = pd.DataFrame({"timestamp": pd.to_datetime(times), "temp_f": temps})
    df["point"] = label
    df = df.dropna()
    return df


def main() -> None:
    dfs: list[pd.DataFrame] = []

    for lat, lon, label in POINTS:
        print(f"Fetching {label} ({lat:.4f}, {lon:.4f}) ...")
        dfs.append(fetch_point(lat, lon, label))
        time.sleep(0.2)  # be polite

    weather = pd.concat(dfs, ignore_index=True)
    if weather.empty:
        raise RuntimeError("Open-Meteo returned no rows. Check internet or date range.")

    # ERCOT-wide hourly aggregation
    agg = (
        weather.groupby("timestamp", as_index=False)
        .agg(
            temp_mean_f=("temp_f", "mean"),
            temp_max_f=("temp_f", "max"),
            temp_min_f=("temp_f", "min"),
            temp_std_f=("temp_f", "std"),
        )
        .sort_values("timestamp")
    )

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/ercot_weather_hourly.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(agg):,} hourly rows")


if __name__ == "__main__":
    main()
