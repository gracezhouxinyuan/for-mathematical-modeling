from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "lstm_series_demo.csv"
METRICS = ROOT / "dataset" / "results_baseline_metrics.csv"
PRED = ROOT / "dataset" / "results_baseline_predictions.csv"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class LSTMRegressor(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def make_windows(values, window=8):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    return np.array(X, dtype="float32")[:, :, None], np.array(y, dtype="float32")


def train_model(model, x_train, y_train, epochs=70):
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()


def main():
    set_seed()
    df = pd.read_csv(DATA)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["value"]]).ravel()
    X, y = make_windows(scaled, window=8)
    split = int(len(X) * 0.75)
    x_train, y_train = torch.tensor(X[:split]), torch.tensor(y[:split])
    x_test, y_test = torch.tensor(X[split:]), y[split:]
    model = LSTMRegressor()
    train_model(model, x_train, y_train)
    with torch.no_grad():
        pred_scaled = model(x_test).numpy()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    truth = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    metrics = pd.DataFrame([{"model": "baseline_lstm", "mae": mean_absolute_error(truth, pred)}])
    metrics.to_csv(METRICS, index=False)
    pd.DataFrame({"y_true": truth, "y_pred": pred}).to_csv(PRED, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
