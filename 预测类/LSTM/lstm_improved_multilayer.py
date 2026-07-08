from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "lstm_series_demo.csv"
METRICS = ROOT / "dataset" / "results_improved_metrics.csv"
PRED = ROOT / "dataset" / "results_improved_predictions.csv"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class ResidualLSTM(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def build_residual_dataset(values, period=24):
    train_cut = int(len(values) * 0.75)
    seasonal_step = np.mean(values[period:train_cut] - values[: train_cut - period])
    target_indices = np.arange(period, len(values))
    seasonal_pred = values[target_indices - period] + seasonal_step
    residual_target = values[target_indices] - seasonal_pred
    return target_indices, seasonal_pred, residual_target, train_cut


def make_windows(scaled_values, target_indices, period=24):
    windows = []
    for idx in target_indices:
        windows.append(scaled_values[idx - period : idx])
    return np.array(windows, dtype="float32")[:, :, None]


def train_model(model, x_train, y_train, epochs=120):
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()


def main():
    set_seed()
    df = pd.read_csv(DATA)
    values = df["value"].to_numpy(dtype="float32")
    target_indices, seasonal_pred, residual_target, train_cut = build_residual_dataset(values)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values[:train_cut].reshape(-1, 1)).ravel()
    full_scaled = scaler.transform(values.reshape(-1, 1)).ravel()
    X = make_windows(full_scaled, target_indices)

    train_mask = target_indices < train_cut
    x_train = torch.tensor(X[train_mask])
    y_train = torch.tensor(residual_target[train_mask].astype("float32"))
    x_test = torch.tensor(X[~train_mask])

    model = ResidualLSTM()
    train_model(model, x_train, y_train)
    with torch.no_grad():
        residual_pred = model(x_test).numpy()

    truth = values[target_indices[~train_mask]]
    pred = seasonal_pred[~train_mask] + residual_pred
    metrics = pd.DataFrame([{"model": "seasonal_residual_lstm", "mae": mean_absolute_error(truth, pred)}])
    metrics.to_csv(METRICS, index=False)
    pd.DataFrame({"y_true": truth, "y_pred": pred}).to_csv(PRED, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
