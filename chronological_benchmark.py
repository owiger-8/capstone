"""
Chronological Evaluation (Time-Series Split)
==========================================
Trains on early data (first 80% chronologically), tests on later data (last 20% chronologically).
This is the standard time-series approach that shows real architectural differences.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

from src.model import CNNGRUFeatureExtractor
from src.edge import EdgePredictor

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 60

class CNN_GRU_Hasanat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        xp = x.permute(0, 2, 1)
        f = self.relu(self.conv(xp)).permute(0, 2, 1)
        g, _ = self.gru(f)
        return self.fc(g[:, -1, :]).squeeze(-1)

class CNN_BiGRU_Attention(nn.Module):
    def __init__(self, gru_hidden=32):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.bigru = nn.GRU(16, gru_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(gru_hidden * 2, 1)
        self.fc = nn.Linear(gru_hidden * 2, 1)
    def forward(self, x):
        xp = x.permute(0, 2, 1)
        f = self.relu(self.conv(xp)).permute(0, 2, 1)
        g, _ = self.bigru(f)
        a = torch.softmax(self.attn(g).squeeze(-1), dim=1).unsqueeze(-1)
        ctx = (g * a).sum(1)
        return self.fc(ctx).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)

class ELM(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(SEQ_LEN, hidden)
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, -1, 1)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        h = torch.relu(self.fc1(x.squeeze(-1)))
        return self.fc2(h).squeeze(-1)

def train_model(model, X_tr, y_tr, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=64, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb.squeeze(-1))
            loss.backward()
            opt.step()
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="household_data_1min_singleindex.csv")
    ap.add_argument("--target", default="DE_KN_residential1_grid_import")
    ap.add_argument("--train_split", type=float, default=0.8)
    args = ap.parse_args()
    
    print("Loading data...")
    df = pd.read_csv(args.csv, usecols=[args.target], nrows=100000)
    df[args.target] = df[args.target].ffill().bfill().fillna(0.0)
    series = df[args.target].values.astype(np.float32)
    
    mn, mx = series.min(), series.max()
    scaled = (series - mn) / (mx - mn + 1e-8)
    
    # Create sequences
    X_all, y_all = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X_all.append(scaled[i:i + SEQ_LEN])
        y_all.append(scaled[i + SEQ_LEN])
    X_all = np.array(X_all)[:, :, None]
    y_all = np.array(y_all)[:, None]
    
    # Chronological split (NO SHUFFLE)
    split = int(args.train_split * len(X_all))
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]
    
    print(f"Training on {len(X_tr)} samples, testing on {len(X_te)} samples (chronological)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train static models
    print("Training CNN-GRU (Hasanat)...")
    cnn_gru = train_model(CNN_GRU_Hasanat(), X_tr, y_tr, epochs=3)
    
    print("Training CNN-BiGRU+Att (Ullah)...")
    cnn_bigru = train_model(CNN_BiGRU_Attention(), X_tr, y_tr, epochs=3)
    
    print("Training LSTM (baseline)...")
    lstm = train_model(LSTMModel(), X_tr, y_tr, epochs=3)
    
    print("Training BiLSTM (Bohara)...")
    bilstm = train_model(BiLSTMModel(), X_tr, y_tr, epochs=3)
    
    print("Training ELM (Yang)...")
    elm = train_model(ELM(), X_tr, y_tr, epochs=3)
    
    def eval_static(model, X_te, y_te):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_te), 4096):
                preds.append(model(torch.from_numpy(X_te[i:i+4096]).float().to(device)).cpu().numpy())
            preds = np.concatenate(preds)
        preds = preds * (mx - mn) + mn
        true = y_te.flatten() * (mx - mn) + mn
        return {
            'mae': mean_absolute_error(true, preds),
            'rmse': mean_squared_error(true, preds) ** 0.5,
            'r2': r2_score(true, preds)
        }
    
    results = {}
    for name, model in [
        ("CNN-GRU (Hasanat2024)", cnn_gru),
        ("CNN-BiGRU+Att (Ullah2025)", cnn_bigru),
        ("LSTM (baseline)", lstm),
        ("BiLSTM (Bohara2023)", bilstm),
        ("ELM (Yang2024)", elm),
    ]:
        results[name] = eval_static(model, X_te, y_te)
        print(f"  {name}: MAE={results[name]['mae']:.4f} RMSE={results[name]['rmse']:.4f} R2={results[name]['r2']:.4f}")
    
    # EcoEdge with online adaptation (use evaluate.py approach)
    print("\nEvaluating EcoEdge (with online adaptation)...")
    ep = EdgePredictor()
    preds_e, trues_e = [], []
    start_idx = split + 60  # start after training period
    for i in range(start_idx, min(start_idx + 50000, len(series))):
        if i < SEQ_LEN:
            continue
        window = series[i - SEQ_LEN:i]
        y_curr = series[i]
        pred, _ = ep.predict(window, y_curr=y_curr)
        preds_e.append(pred)
        trues_e.append(float(y_curr))
    
    results["EcoEdge (ours, adaptive)"] = {
        'mae': mean_absolute_error(trues_e, preds_e),
        'rmse': mean_squared_error(trues_e, preds_e) ** 0.5,
        'r2': r2_score(trues_e, preds_e)
    }
    print(f"  EcoEdge (ours, adaptive): MAE={results['EcoEdge (ours, adaptive)']['mae']:.4f} RMSE={results['EcoEdge (ours, adaptive)']['rmse']:.4f} R2={results['EcoEdge (ours, adaptive)']['r2']:.4f}")
    
    # Save results
    with open('chronological_benchmark_results.md', 'w') as f:
        f.write("# Chronological Evaluation Results\n\n")
        f.write("**Setup:** Time-series split (chronological, no shuffle). Train on first 80%, test on last 20%. ")
        f.write("Static models are frozen. EcoEdge uses online k-NN memory adaptation.\n\n")
        f.write("## Results (MAE, RMSE, R² — MAPE excluded)\n\n")
        f.write("| Model | Type | MAE | RMSE | R² |\n")
        f.write("|-------|------|-----|------|----|\n")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]['mae']):
            mtype = "Static (frozen)" if "adaptive" not in name.lower() else "Adaptive (k-NN buffer)"
            f.write(f"| {name} | {mtype} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |\n")
    
    print("\nResults saved to chronological_benchmark_results.md")
    print("="*80)
    for name, m in sorted(results.items(), key=lambda kv: kv[1]['mae']):
        print(f"{name:40s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
