import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.edge import EdgePredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="household_data_1min_singleindex.csv")
    ap.add_argument("--target", default="DE_KN_residential1_grid_import")
    ap.add_argument("--start", type=int, default=50000)
    ap.add_argument("--n", type=int, default=300000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=[args.target])
    df[args.target] = df[args.target].ffill().bfill()
    series = np.nan_to_num(df[args.target].values.astype(np.float32), nan=0.0)

    ep = EdgePredictor()
    preds, trues = [], []
    for i in range(args.start, args.start + args.n):
        window = series[i - 60:i]
        y_curr = series[i]
        pred, _ = ep.predict(window, y_curr=y_curr)
        preds.append(pred)
        trues.append(float(y_curr))

    preds = np.nan_to_num(np.array(preds)); trues = np.array(trues)
    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds) ** 0.5
    mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100
    r2 = r2_score(trues, preds)
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}% R2={r2:.4f}")


if __name__ == "__main__":
    main()
