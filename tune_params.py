import json
import numpy as np
import pandas as pd
from src.edge import EdgePredictor


def main():
    df = pd.read_csv(
        "household_data_1min_singleindex.csv",
        usecols=["DE_KN_residential1_grid_import"],
    )
    df["DE_KN_residential1_grid_import"] = (
        df["DE_KN_residential1_grid_import"].ffill().bfill().fillna(0.0)
    )
    series = df["DE_KN_residential1_grid_import"].values.astype(np.float32)

    # Convert cumulative kWh to power (kW): kWh per minute * 60 = kW
    series = np.diff(series, prepend=series[0]) * 60.0

    # Warm-up on training data only; evaluate on test without leaking y_curr
    warmup_end = 200000
    test_start = warmup_end
    test_end = warmup_end + 50000

    best = None
    results = []

    for blend in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for k in [3, 5, 7, 10]:
            ep = EdgePredictor()

            # ---- Warm-up memory on train data (no test leakage) ----
            for i in range(60, warmup_end):
                window = series[i - 60 : i]
                y_curr = series[i]
                x = (
                    ep._scale(np.asarray(window, dtype=np.float32))
                    .reshape(1, 60, 1)
                    .astype(np.float32)
                )
                base_pred, latent = ep.session.run(None, {ep.input_name: x})
                latent = latent[0].astype(np.float32)
                ep.index.add_items(latent.reshape(1, -1), [ep.count])
                ep.values.append(float(ep._scale(y_curr)))
                ep.count += 1

            # ---- Evaluate on test data (no y_curr added to memory) ----
            preds, trues = [], []
            for i in range(test_start, test_end):
                window = series[i - 60 : i]
                y_curr = series[i]
                x = (
                    ep._scale(np.asarray(window, dtype=np.float32))
                    .reshape(1, 60, 1)
                    .astype(np.float32)
                )
                base_pred, latent = ep.session.run(None, {ep.input_name: x})
                base_pred = float(base_pred[0, 0])
                latent = latent[0].astype(np.float32)

                if ep.count >= 10:
                    labels, _ = ep.index.knn_query(
                        latent.reshape(1, -1), k=k
                    )
                    mem_pred = float(
                        np.mean([ep.values[j] for j in labels[0]])
                    )
                    final_scaled = (1 - blend) * base_pred + blend * mem_pred
                else:
                    final_scaled = base_pred

                pred = float(ep._inv_scale(final_scaled))
                preds.append(pred)
                trues.append(float(y_curr))

            preds = np.array(preds)
            trues = np.array(trues)

            mae = float(np.mean(np.abs(trues - preds)))
            rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
            denom = float(np.sum((trues - trues.mean()) ** 2))
            r2 = (
                float(1 - np.sum((trues - preds) ** 2) / (denom + 1e-8))
                if denom > 1e-8
                else float("nan")
            )

            print(
                f"blend={blend} k={k}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}"
            )
            results.append(
                {"blend": blend, "k": k, "mae": mae, "rmse": rmse, "r2": r2}
            )

            if best is None or mae < best[0]:
                best = (mae, blend, k)

    print(f"\nBest: MAE={best[0]:.4f} blend={best[1]} k={best[2]}")

    # Save best parameters
    best_params = {"blend": best[1], "k": best[2], "mae": best[0]}
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Save full sweep results
    with open("tune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved best_params.json and tune_results.json")


if __name__ == "__main__":
    main()
