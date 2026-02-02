from __future__ import annotations

import pandas as pd
import holidays


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # US Federal Holidays
    min_year = int(df["timestamp"].dt.year.min())
    max_year = int(df["timestamp"].dt.year.max())
    us_holidays = holidays.US(years=range(min_year, max_year + 1))

    df["date"] = df["timestamp"].dt.date
    df["holiday_name"] = df["date"].map(us_holidays).fillna("")
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)

    # ============================
    # ERCOT-specific special days
    # ============================

    df["is_ercot_special_day"] = 0

    # Christmas Eve
    df.loc[
        (df["date"].astype(str).str.endswith("-12-24")),
        "is_ercot_special_day",
    ] = 1

    # New Year's Eve
    df.loc[
        (df["date"].astype(str).str.endswith("-12-31")),
        "is_ercot_special_day",
    ] = 1

    # Day after Thanksgiving (Friday)
    # Thanksgiving = 4th Thursday in November
    thanksgiving = holidays.US(years=range(min_year, max_year + 1)).get_named(
        "Thanksgiving"
    )
    day_after_thanksgiving = {d + pd.Timedelta(days=1) for d in thanksgiving}

    df.loc[df["date"].isin(day_after_thanksgiving), "is_ercot_special_day"] = 1

    # July 3rd or 5th if adjacent to July 4th midweek
    df.loc[
        (df["date"].astype(str).str.endswith("-07-03"))
        | (df["date"].astype(str).str.endswith("-07-05")),
        "is_ercot_special_day",
    ] = 1

    return df.drop(columns=["date"])


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp")

    # Hourly lags
    df["lag_24"] = df["load_mw"].shift(24)
    df["lag_48"] = df["load_mw"].shift(48)
    df["lag_72"] = df["load_mw"].shift(72)

    # Weekly lags
    df["lag_168"] = df["load_mw"].shift(168)
    df["lag_336"] = df["load_mw"].shift(336)

    # Rolling stats
    df["roll_mean_24"] = df["load_mw"].shift(1).rolling(24).mean()
    df["roll_mean_168"] = df["load_mw"].shift(1).rolling(168).mean()
    df["roll_std_168"] = df["load_mw"].shift(1).rolling(168).std()

    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaNs introduced by lag/rolling features.
    This should ONLY be done after all features are created.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows due to NaNs from lags/rolling features.")
    return df


def build_day_ahead_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Keep only demand + day-ahead forecast
    df = df[df["type"].isin(["D", "DF"])].copy()

    # Pivot to wide FIRST
    wide = (
        df.pivot_table(
            index="timestamp",
            columns="type",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"D": "load_mw", "DF": "da_forecast_mw"})
        .sort_values("timestamp")
    )

    # Merge weather AFTER pivot
    weather = (
        pd.read_parquet("data/raw/ercot_weather_hourly.parquet")
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
    )

    # Debug: confirm weather file structure
    print("Weather columns:", list(weather.columns))
    print("Weather timestamp sample:", weather["timestamp"].head(3).to_list())

    wide = wide.merge(weather, on="timestamp", how="left")
    max_weather_ts = weather["timestamp"].max()
    wide = wide[wide["timestamp"] <= max_weather_ts].copy()

    # Debug: confirm merge worked
    print("Merged temp_mean_f null rate:", wide["temp_mean_f"].isna().mean())

    # -----------------------------
    # Weather features (Fahrenheit)
    # -----------------------------
    wide["cdh_65f"] = (wide["temp_mean_f"] - 65.0).clip(lower=0)
    wide["hdh_65f"] = (65.0 - wide["temp_mean_f"]).clip(lower=0)

    for lag in [1, 6, 12, 24]:
        wide[f"temp_mean_f_lag_{lag}"] = wide["temp_mean_f"].shift(lag)

    wide["temp_mean_f_roll_mean_24"] = wide["temp_mean_f"].rolling(24).mean()
    wide["temp_mean_f_roll_std_24"] = wide["temp_mean_f"].rolling(24).std()

    # Calendar + load lags
    wide = add_calendar_features(wide)
    wide = add_lag_features(wide)

    # Drop NaNs ONCE at the end (includes weather lags + load lags)
    wide = drop_na_rows(wide)

    return wide.sort_values("timestamp")
    wide = drop_na_rows(wide)

    return wide.sort_values("timestamp")


if __name__ == "__main__":
    df = build_day_ahead_dataset("data/processed/eia930_erco_tidy.parquet")

    # Drop rows with any missing values after feature creation
    df = df.dropna().reset_index(drop=True)

    print(df.head())
    print("\nHoliday count:", df["is_holiday"].sum())

    df.to_parquet(
        "data/processed/ercot_day_ahead_dataset.parquet",
        index=False,
    )
