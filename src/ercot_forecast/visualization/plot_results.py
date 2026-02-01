from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def _savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    df = pd.read_parquet("reports/preds_all.parquet").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ----- Full view (may be dense) -----
    plt.figure()
    plt.plot(df["timestamp"], df["load_mw"], label="Actual")
    plt.plot(df["timestamp"], df["yhat_df"], label="DF baseline")
    plt.plot(df["timestamp"], df["yhat_xgb"], label="XGB_resid")
    plt.plot(df["timestamp"], df["yhat_lstm"], label="LSTM_resid")
    plt.legend()
    plt.title("ERCOT Day-ahead Forecasts (Test Period)")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    _savefig("reports/figures/forecasts_full.png")

    # ----- Last 30 days -----
    end = df["timestamp"].max()
    start = end - pd.Timedelta(days=30)
    z = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

    plt.figure()
    plt.plot(z["timestamp"], z["load_mw"], label="Actual")
    plt.plot(z["timestamp"], z["yhat_df"], label="DF baseline")
    plt.plot(z["timestamp"], z["yhat_xgb"], label="XGB_resid")
    plt.plot(z["timestamp"], z["yhat_lstm"], label="LSTM_resid")
    plt.legend()
    plt.title("ERCOT Day-ahead Forecasts (Last 30 days of Test)")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    _savefig("reports/figures/forecasts_last30d.png")

    # ----- Error distribution -----
    df["err_df"] = df["yhat_df"] - df["load_mw"]
    df["err_xgb"] = df["yhat_xgb"] - df["load_mw"]
    df["err_lstm"] = df["yhat_lstm"] - df["load_mw"]

    plt.figure()
    plt.hist(df["err_df"], bins=80, alpha=0.6, label="DF")
    plt.hist(df["err_xgb"], bins=80, alpha=0.6, label="XGB_resid")
    plt.hist(df["err_lstm"], bins=80, alpha=0.6, label="LSTM_resid")
    plt.legend()
    plt.title("Forecast Error Distribution (yhat - actual)")
    plt.xlabel("Error (MW)")
    plt.ylabel("Count")
    _savefig("reports/figures/error_hist.png")

    # ----- Peak-focused plot -----
    # Define peak hours as top 5% actual load in test
    thresh = df["load_mw"].quantile(0.95)
    peaks = df[df["load_mw"] >= thresh].copy()

    plt.figure()
    plt.scatter(
        peaks["load_mw"],
        (peaks["yhat_df"] - peaks["load_mw"]).abs(),
        s=8,
        label="DF",
        alpha=0.6,
    )
    plt.scatter(
        peaks["load_mw"],
        (peaks["yhat_xgb"] - peaks["load_mw"]).abs(),
        s=8,
        label="XGB_resid",
        alpha=0.6,
    )
    plt.scatter(
        peaks["load_mw"],
        (peaks["yhat_lstm"] - peaks["load_mw"]).abs(),
        s=8,
        label="LSTM_resid",
        alpha=0.6,
    )
    plt.legend()
    plt.title("Absolute Error on Peak Hours (Top 5% load)")
    plt.xlabel("Actual load (MW)")
    plt.ylabel("Absolute error (MW)")
    _savefig("reports/figures/peak_abs_error.png")


if __name__ == "__main__":
    main()
