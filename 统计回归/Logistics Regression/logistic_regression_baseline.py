from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'logistic_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_baseline_metrics.csv'
PRED = ROOT / 'dataset' / 'results_baseline_predictions.csv'


def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=['y'])
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = pd.DataFrame([{
        'model': 'baseline',
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'auc': roc_auc_score(y_test, proba),
        'tn_fp_fn_tp': str(confusion_matrix(y_test, pred).ravel().tolist()),
    }])
    metrics.to_csv(OUT, index=False)
    pd.DataFrame({'y_true': y_test.to_numpy(), 'proba': proba, 'pred': pred}).to_csv(PRED, index=False)
    print(metrics)


if __name__ == '__main__':
    main()
