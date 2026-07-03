from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'kalman_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_improved_state.csv'


def adaptive_kalman(measurements, dt=1.0, process_var=0.05, measure_var=1.2):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q_base = process_var * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    x = np.array([[measurements[0]], [0.0]])
    P = np.eye(2) * 10.0
    R = np.array([[measure_var]])
    estimates = []
    innovation_window = []
    for z in measurements:
        x = F @ x
        P = F @ P @ F.T + Q_base
        y = np.array([[z]]) - H @ x
        innovation_window.append(float(y**2))
        if len(innovation_window) > 8:
            innovation_window.pop(0)
        adaptive_r = max(np.mean(innovation_window), 0.2)
        R = np.array([[adaptive_r]])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        estimates.append(x.ravel())
    return np.array(estimates)


def main() -> None:
    df = pd.read_csv(DATA)
    est = adaptive_kalman(df['measurement'].to_numpy())
    out = pd.DataFrame({'t': df['t'], 'est_pos': est[:, 0], 'est_vel': est[:, 1], 'true_pos': df['true_pos'], 'measurement': df['measurement']})
    out.to_csv(OUT, index=False)
    print(out.head())


if __name__ == '__main__':
    main()
