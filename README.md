# ERCOT Day-Ahead Load Forecasting

Forecast ERCOT hourly demand for the next 24 hours using:
- **Baseline**: EIA-930 day-ahead demand forecast (DF)
- **Residual XGBoost**: predict the error of DF and correct it
- **Residual LSTM**: sequence model that learns DF residual structure (uses Apple Silicon MPS)

This repo is designed to be **reproducible**: clean data pipeline, feature engineering, comparable metrics, and plots.

---

## Results (test set: >= 2023-01-01)

Your current run (common timestamps) shows:

| Model | MAE | RMSE | MAPE (%) | Peak MAPE (%) |
|------:|----:|-----:|---------:|--------------:|
| DF baseline | 1221 | 1714 | 2.38 | 1.60 |
| XGBoost (residual) | 1264 | 1715 | 2.41 | 2.13 |
| LSTM (residual) | 1426 | 1988 | 2.78 | 1.74 |

> Note: DF is an extremely strong baseline. Residual models can still help on specific regimes (ramps / peaks).  
> See plots in `reports/` (not committed).

---

## Project structure


