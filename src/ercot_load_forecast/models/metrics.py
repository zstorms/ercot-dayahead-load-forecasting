from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    denom = np.where(np.abs(y) < 1e-6, np.nan, np.abs(y))
    return float(np.nanmean(np.abs((y - yhat) / denom)) * 100.0)


def peak_mape(y, yhat, top_pct: float = 0.05) -> float:
    """
    MAPE on top X% highest-load hours by actual load.
    top_pct=0.05 => top 5% load hours.
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    thr = np.quantile(y, 1 - top_pct)
    mask = y >= thr
    if mask.sum() == 0:
        return float("nan")
    denom = np.where(np.abs(y[mask]) < 1e-6, np.nan, np.abs(y[mask]))
    return float(np.nanmean(np.abs((y[mask] - yhat[mask]) / denom)) * 100.0)


def evaluate_all(y, yhat) -> dict[str, float]:
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE_%": mape(y, yhat),
        "Peak_MAPE_%": peak_mape(y, yhat),
    }
