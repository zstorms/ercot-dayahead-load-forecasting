from __future__ import annotations

import os
import pandas as pd
from xgboost import XGBRegressor

from src.ercot_load_forecast.models.metrics import evaluate_all


FEATURES = [
    # published day-ahead forecast (keep as a feature even though we add it back)
    "da_forecast_mw",
    # calendar
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "is_ercot_special_day",
    # lags/rolling
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "lag_336",
    "roll_mean_24",
    "roll_mean_168",
    "roll_std_168",
    # weather
    "temp_mean_f",
    "temp_max_f",
    "temp_min_f",
    "temp_std_f",
    "cdh_65f",
    "hdh_65f",
    "temp_mean_f_lag_1",
    "temp_mean_f_lag_6",
    "temp_mean_f_lag_12",
    "temp_mean_f_lag_24",
    "temp_mean_f_roll_mean_24",
    "temp_mean_f_roll_std_24",
]


def main() -> None:
    df = pd.read_parquet("data/processed/ercot_day_ahead_dataset.parquet").sort_values(
        "timestamp"
    )

    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["timestamp"] < cutoff].copy()
    test = df[df["timestamp"] >= cutoff].copy()

    # -------------------------
    # Residual target
    # resid = actual - DF
    # -------------------------
    train["resid"] = train["load_mw"] - train["da_forecast_mw"]
    test["resid"] = test["load_mw"] - test["da_forecast_mw"]

    X_train, y_train = train[FEATURES], train["resid"]
    X_test = test[FEATURES]

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=0,
    )

    model.fit(X_train, y_train)

    # -------------------------
    # Predictions (MW)
    # -------------------------
    test = test.copy()
    test["yhat_df"] = test["da_forecast_mw"]

    # predict residuals, then add DF back
    test["resid_hat_xgb"] = model.predict(X_test)
    test["yhat_xgb"] = test["da_forecast_mw"] + test["resid_hat_xgb"]

    # -------------------------
    # Metrics
    # -------------------------
    rows = []
    for name, col in [("DF baseline", "yhat_df"), ("XGBoost_resid", "yhat_xgb")]:
        m = evaluate_all(test["load_mw"], test[col])
        m["model"] = name
        rows.append(m)

    out = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    print("\nModel comparison:")
    print(out)

    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/xgb_resid_metrics.csv")
    print("\nSaved: reports/xgb_resid_metrics.csv")

    test[
        [
            "timestamp",
            "load_mw",
            "da_forecast_mw",
            "yhat_df",
            "yhat_xgb",
            "resid_hat_xgb",
        ]
    ].to_parquet("reports/preds_xgb_resid.parquet", index=False)

    print("\nSaved: reports/preds_xgb_resid.parquet")


if __name__ == "__main__":
    main()
