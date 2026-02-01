from __future__ import annotations

import os
import requests
import pandas as pd

# Major TX airport stations (good ERCOT coverage)
STATIONS = [
    "USW00012960",  # Houston
    "USW00013958",  # Dallas-Fort Worth
    "USW00012921",  # Austin
    "USW00012918",  # San Antonio
    "USW00012924",  # Corpus Christi
    "USW00003927",  # El Paso
    "USW00012919",  # Brownsville
    "USW00012970",  # Lubbock
]

START = "2018-01-01"
END = "2024-12-31"

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_station(station: str) -> pd.DataFrame:
    params = {
        "dataset": "global-hourly",
        "stations": station,
        "startDate": START,
        "endDate": END,
        "dataTypes": "TMP",
        "format": "json",
        "units": "metric",
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    # If NOAA returns no rows, skip this station cleanly
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "temp_f", "station"])

    # Validate expected columns exist
    if "DATE" not in df.columns or "TMP" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "temp_f", "station"])

    df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce")

    tmp = df["TMP"].astype(str).str.split(",").str[0]
    tmp = pd.to_numeric(tmp, errors="coerce")  # C in numeric, NaN if missing

    df["temp_f"] = tmp * 9 / 5 + 32
    df["station"] = station

    df = df.dropna(subset=["timestamp", "temp_f"])
    return df[["timestamp", "temp_f", "station"]]


def main():
    dfs = []
    for s in STATIONS:
        print(f"Fetching {s}...")
        d = fetch_station(s)
        if not d.empty:
            dfs.append(d)

    if not dfs:
        raise RuntimeError(
            "No weather rows returned from NOAA for any station. "
            "Check station IDs/date range or NOAA service availability."
        )

    weather = pd.concat(dfs, ignore_index=True)

    print("Weather rows:", len(weather), "Columns:", list(weather.columns))
    print(weather.head())

    # ERCOT-wide hourly aggregation
    agg = (
        weather.groupby("timestamp")
        .agg(
            temp_mean_f=("temp_f", "mean"),
            temp_max_f=("temp_f", "max"),
            temp_min_f=("temp_f", "min"),
            temp_std_f=("temp_f", "std"),
        )
        .reset_index()
    )

    os.makedirs("data/raw", exist_ok=True)
    agg.to_parquet("data/raw/ercot_weather_hourly.parquet", index=False)

    print("Saved data/raw/ercot_weather_hourly.parquet")


if __name__ == "__main__":
    main()
