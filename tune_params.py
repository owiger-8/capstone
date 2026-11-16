import numpy as np
import pandas as pd
from src.edge import EdgePredictor

df = pd.read_csv("household_data_1min_singleindex.csv", usecols=["DE_KN_residential1_grid_import"])
df["DE_KN_residential1_grid_import"] = df["DE_KN_residential1_grid_import"].ffill().bfill().fillna(0.0)
series = df["DE_KN_residential1_grid_import"].values.astype(np.float32)

best = None
for blend in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for k in [3, 5, 7, 10]:
        ep = EdgePredictor()
        ep.index = ep.index  # keep default
        # Override predict to use custom blend and k
        orig_predict = ep.predict
        preds, trues = [], []
        for i in range(200000, 200000 + 50000):
            window = series[i - 60:i]
            y_curr = series[i]
            x = ep._scale(np.asarray(window, dtype=np.float32)).reshape(1, 60, 1).astype(np.float32)
            base_pred, latent = ep.session.run(None, {ep.input_name: x})
            base_pred = float(base_pred[0, 0])
            latent = latent[0].astype(np.float32)
            if ep.count >= 10:
                labels, _ = ep.index.knn_query(latent.reshape(1, -1), k=k)
                mem_pred = float(np.mean([ep.values[i] for i in labels[0]]))
                final_scaled = (1-blend) * base_pred + blend * mem_pred
            else:
                final_scaled = base_pred
            ep.index.add_items(latent.reshape(1, -1), [ep.count])
            ep.values.append(float(ep._scale(y_curr)))
            ep.count += 1
            pred = float(ep._inv_scale(final_scaled))
            preds.append(pred)
            trues.append(float(y_curr))
        preds = np.array(preds); trues = np.array(trues)
        mae = np.mean(np.abs(trues - preds))
        rmse = np.sqrt(np.mean((trues - preds)**2))
        r2 = 1 - np.sum((trues - preds)**2) / np.sum((trues - trues.mean())**2)
        print(f"blend={blend} k={k}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
        if best is None or mae < best[0]:
            best = (mae, blend, k)
print(f"\nBest: MAE={best[0]:.4f} blend={best[1]} k={best[2]}")
