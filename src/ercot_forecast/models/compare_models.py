from __future__ import annotations

import os
import pandas as pd

from src.ercot_forecast.models.metrics import evaluate_all


def main() -> None:
    # Load predictions
    xgb = pd.read_parquet("reports/preds_xgb_resid.parquet")
    lstm = pd.read_parquet("reports/preds_lstm_resid.parquet")

    # Normalize timestamp types
    xgb["timestamp"] = pd.to_datetime(xgb["timestamp"])
    lstm["timestamp"] = pd.to_datetime(lstm["timestamp"])

    # Keep only what we need
    xgb = xgb[["timestamp", "load_mw", "da_forecast_mw", "yhat_df", "yhat_xgb"]].copy()
    lstm = lstm[["timestamp", "yhat_lstm"]].copy()

    # Merge (inner join ensures same timestamps)
    df = xgb.merge(lstm, on="timestamp", how="inner")

    # Sanity checks
    if df.empty:
        raise RuntimeError(
            "Merged comparison dataframe is empty. Check that both preds files overlap in timestamps."
        )

    # Compute metrics
    rows = []
    for name, col in [
        ("DF baseline", "yhat_df"),
        ("XGBoost_resid", "yhat_xgb"),
        ("LSTM_resid", "yhat_lstm"),
    ]:
        m = evaluate_all(df["load_mw"], df[col])
        m["model"] = name
        rows.append(m)

    out = pd.DataFrame(rows).set_index("model").sort_values("MAE")

    print("\nModel comparison (common timestamps):")
    print(out)

    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/metrics_all.csv")
    df.to_parquet("reports/preds_all.parquet", index=False)

    print("\nSaved: reports/metrics_all.csv")
    print("Saved: reports/preds_all.parquet")


if __name__ == "__main__":
    main()
