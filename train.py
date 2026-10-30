import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model import CNNGRUFeatureExtractor


def make_sequences(series, seq_len=60):
    vals = np.nan_to_num(series.astype(np.float32).values if hasattr(series, "values") else np.asarray(series, dtype=np.float32), nan=0.0)
    mn, mx = vals.min(), vals.max()
    scaled = (vals - mn) / (mx - mn + 1e-8)
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len])
    X = np.array(X)[:, :, None]
    y = np.array(y)[:, None]
    return X, y, float(mn), float(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="household_data_1min_singleindex.csv")
    ap.add_argument("--target", default="DE_KN_residential1_grid_import")
    ap.add_argument("--rows", type=int, default=2000000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="cnn_gru_base.onnx")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=[args.target], nrows=args.rows)
    df[args.target] = df[args.target].ffill().bfill().fillna(0.0)
    X, y, mn, mx = make_sequences(df[args.target], 60)

    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNGRUFeatureExtractor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(loader):.6f}")

    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(Xte), 64):
            xb = torch.from_numpy(Xte[i:i+64]).to(device)
            p, _ = model(xb)
            preds_list.append(p.cpu().numpy())
    val_pred = np.concatenate(preds_list, axis=0)
    val_loss = loss_fn(torch.from_numpy(val_pred), torch.from_numpy(yte)).item()
    val_r2 = 1 - ((yte - val_pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    print(f"val_loss={val_loss:.6f} val_r2={float(val_r2):.4f}")
    np.save("metrics.npy", np.array([total / len(loader), val_loss, float(val_r2)]))

    dummy = torch.zeros(1, 60, 1, device=device)
    model.cpu()
    torch.onnx.export(model, dummy.cpu(), args.out, input_names=["input"], output_names=["prediction", "latent"],
                      dynamic_axes={"input": {0: "batch"}, "prediction": {0: "batch"}, "latent": {0: "batch"}})
    np.save("scaler.npy", np.array([mn, mx]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
