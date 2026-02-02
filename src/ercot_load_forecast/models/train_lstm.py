from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as L

from src.ercot_load_forecast.models.metrics import evaluate_all


# ============================
# Config
# ============================
CUTOFF = pd.Timestamp("2023-01-01")
PAST_H = 168
HORIZON = 24

FEATURES = [
    # scaled history signal
    "load_mw_z",
    # continuous
    "da_forecast_mw",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "lag_336",
    "roll_mean_24",
    "roll_mean_168",
    "roll_std_168",
    # calendar (kept numeric as-is)
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "is_ercot_special_day",
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

# Standardize these (fit on TRAIN only)
CONTINUOUS = [
    "da_forecast_mw",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "lag_336",
    "roll_mean_24",
    "roll_mean_168",
    "roll_std_168",
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


# ============================
# Utils
# ============================
def fit_standardizer(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = df[cols].mean()
    std = df[cols].std().replace(0, 1.0)
    return mean, std


def apply_standardizer(
    df: pd.DataFrame, cols: list[str], mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (out[cols] - mean) / std
    return out


# ============================
# Dataset
# ============================
class WindowDataset(Dataset):
    """
    Returns:
      X: (PAST_H, n_features)
      y: (HORIZON,)   # scaled residual
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        self.df = df.reset_index(drop=True)
        self.X = self.df[feature_cols].to_numpy(np.float32)
        self.y = self.df["y_scaled"].to_numpy(np.float32)

        self.n = len(self.df) - (PAST_H + HORIZON) + 1
        if self.n <= 0:
            raise ValueError("Not enough rows for chosen PAST_H and HORIZON.")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + PAST_H]
        y = self.y[idx + PAST_H : idx + PAST_H + HORIZON]
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================
# Model
# ============================
class LSTMForecaster(L.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, HORIZON),
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

    def training_step(self, batch, _):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("train_mae", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("val_mae", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


# ============================
# Prediction helper
# ============================
@torch.no_grad()
def predict_next24_resid(
    model: LSTMForecaster,
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    features: list[str],
    device: str,
    resid_mean: float,
    resid_std: float,
) -> pd.DataFrame:
    """
    Produces MW predictions:
      yhat_mw = da_forecast_mw_raw + resid_hat_mw
    """
    model.eval().to(device)

    X = df_scaled[features].to_numpy(np.float32)
    ts = df_raw["timestamp"].to_numpy()
    da_raw = df_raw["da_forecast_mw"].to_numpy(dtype=np.float64)

    rows: list[tuple[np.datetime64, float]] = []
    n = len(df_scaled) - (PAST_H + HORIZON) + 1

    for i in range(n):
        x = torch.from_numpy(X[i : i + PAST_H]).unsqueeze(0).to(device)
        resid_hat_scaled = model(x).squeeze(0).cpu().numpy()  # (HORIZON,)
        resid_hat = resid_hat_scaled * resid_std + resid_mean  # MW residual

        start = i + PAST_H
        for h in range(HORIZON):
            yhat_mw = float(da_raw[start + h] + resid_hat[h])
            rows.append((ts[start + h], yhat_mw))

    pred = pd.DataFrame(rows, columns=["timestamp", "yhat_lstm"])
    pred = pred.groupby("timestamp", as_index=False)["yhat_lstm"].mean()
    return pred


# ============================
# Main
# ============================
def main() -> None:
    df = pd.read_parquet("data/processed/ercot_day_ahead_dataset.parquet").sort_values(
        "timestamp"
    )

    # Keep a raw copy for evaluation/baseline addition
    raw = df[["timestamp", "load_mw", "da_forecast_mw"]].copy()

    train = df[df["timestamp"] < CUTOFF].copy()
    test = df[df["timestamp"] >= CUTOFF].copy()

    # ---- target = residual (MW) ----
    train["resid"] = train["load_mw"] - train["da_forecast_mw"]
    test["resid"] = test["load_mw"] - test["da_forecast_mw"]

    resid_mean = float(train["resid"].mean())
    resid_std = float(train["resid"].std()) if float(train["resid"].std()) != 0 else 1.0

    train["y_scaled"] = (train["resid"] - resid_mean) / resid_std
    test["y_scaled"] = (test["resid"] - resid_mean) / resid_std

    # ---- input history signal: z-scored load ----
    load_mean = float(train["load_mw"].mean())
    load_std = (
        float(train["load_mw"].std()) if float(train["load_mw"].std()) != 0 else 1.0
    )

    train["load_mw_z"] = (train["load_mw"] - load_mean) / load_std
    test["load_mw_z"] = (test["load_mw"] - load_mean) / load_std

    # ---- scale continuous covariates (fit on TRAIN only) ----
    x_mean, x_std = fit_standardizer(train, CONTINUOUS)
    train = apply_standardizer(train, CONTINUOUS, x_mean, x_std)
    test = apply_standardizer(test, CONTINUOUS, x_mean, x_std)

    # Datasets / loaders
    train_ds = WindowDataset(train, FEATURES)
    val_ds = WindowDataset(test, FEATURES)

    train_dl = DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_dl = DataLoader(val_ds, batch_size=256, num_workers=4, persistent_workers=True)

    model = LSTMForecaster(n_features=len(FEATURES))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=10,
        accelerator=device,
        devices=1,
        default_root_dir="reports/lstm_resid",
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dl, val_dl)

    # ---- Evaluation (MW) ----
    preds = predict_next24_resid(
        model=model,
        df_scaled=test,
        df_raw=raw[raw["timestamp"] >= CUTOFF].reset_index(drop=True),
        features=FEATURES,
        device=device,
        resid_mean=resid_mean,
        resid_std=resid_std,
    )

    truth = raw[raw["timestamp"] >= CUTOFF].copy()
    truth["yhat_df"] = truth["da_forecast_mw"]

    merged = truth.merge(preds, on="timestamp", how="inner")

    rows = []
    for name, col in [("DF", "yhat_df"), ("LSTM_resid", "yhat_lstm")]:
        m = evaluate_all(merged["load_mw"], merged[col])
        m["model"] = name
        rows.append(m)

    out = pd.DataFrame(rows).set_index("model")
    print(out)

    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/lstm_resid_metrics.csv")
    merged.to_parquet("reports/preds_lstm_resid.parquet", index=False)
    print(
        "\nSaved: reports/lstm_resid_metrics.csv and reports/preds_lstm_resid.parquet"
    )


if __name__ == "__main__":
    main()
