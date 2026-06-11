from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'kalman_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_baseline_state.csv'


def kalman_filter(measurements, dt=1.0, process_var=0.08, measure_var=1.44):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = process_var * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    R = np.array([[measure_var]])
    x = np.array([[measurements[0]], [0.0]])
    P = np.eye(2) * 10.0
    estimates = []
    for z in measurements:
        x = F @ x
        P = F @ P @ F.T + Q
        y = np.array([[z]]) - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        estimates.append(x.ravel())
    return np.array(estimates)


def main() -> None:
    df = pd.read_csv(DATA)
    est = kalman_filter(df['measurement'].to_numpy())
    out = pd.DataFrame({'t': df['t'], 'est_pos': est[:, 0], 'est_vel': est[:, 1], 'true_pos': df['true_pos'], 'measurement': df['measurement']})
    out.to_csv(OUT, index=False)
    print(out.head())


if __name__ == '__main__':
    main()
