import time, psutil, numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.edge import EdgePredictor

def run(csv, target, start=50000, n=500):
    df = pd.read_csv(csv, usecols=[target])
    df[target] = df[target].ffill().bfill().fillna(0.0)
    series = df[target].values.astype(np.float32)
    ep = EdgePredictor()
    preds, trues = [], []
    fig, ax = plt.subplots()
    for i in range(start, start + n):
        window = series[i-60:i]
        y = float(series[i])
        pred, _ = ep.predict(window, y_curr=y)
        preds.append(pred); trues.append(y)
        if i % 20 == 0:
            ax.clear()
            ax.plot(trues[-100:], label="actual")
            ax.plot(preds[-100:], label="pred")
            ax.set_title(f"MAE={np.mean(np.abs(np.array(preds)-np.array(trues))):.3f} CPU={psutil.cpu_percent()}% MEM={psutil.virtual_memory().percent}%")
            ax.legend()
            plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    run("household_data_1min_singleindex.csv", "DE_KN_residential1_grid_import")
