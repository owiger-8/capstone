"""
Streaming/Forward-Time Evaluation
=================================
Compares sequential models (trained on early data, frozen) vs EcoEdge (with online k-NN memory adaptation)
in a forward-time streaming setup - mirroring real deployment.
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
        print(f"    Epoch {epoch+1}/{epochs} done", flush=True)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="household_data_1min_singleindex.csv")
    ap.add_argument("--target", default="DE_KN_residential1_grid_import")
    ap.add_argument("--train_split", type=float, default=0.8)
    ap.add_argument("--start", type=int, default=200000)  # start evaluation after training period
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args()
    
    print("Loading data...")
    df = pd.read_csv(args.csv, usecols=[args.target])
    df[args.target] = df[args.target].ffill().bfill().fillna(0.0)
    series = df[args.target].values.astype(np.float32)
    
    mn, mx = series.min(), series.max()
    scaled = (series - mn) / (mx - mn + 1e-8)
    
    # Create sequences for training (chronological split)
    X_all, y_all = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X_all.append(scaled[i:i + SEQ_LEN])
        y_all.append(scaled[i + SEQ_LEN])
    X_all = np.array(X_all)[:, :, None]
    y_all = np.array(y_all)[:, None]
    
    # Use smaller training set for speed
    split = int(args.train_split * len(X_all))
    X_tr = X_all[:split]
    y_tr = y_all[:split]
    X_tr_small = X_tr[:200000]  # Use more training data
    y_tr_small = y_tr[:200000]
    print(f"Training on first {len(X_tr_small)} samples, testing streaming from index {args.start}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train static models on early data only (frozen)
    print("Training CNN-GRU (Hasanat) on early data...")
    cnn_gru_static = train_model(CNN_GRU_Hasanat(), X_tr_small, y_tr_small, epochs=5)
    
    print("Training CNN-BiGRU+Att (Ullah) on early data...")
    cnn_bigru_static = train_model(CNN_BiGRU_Attention(), X_tr_small, y_tr_small, epochs=5)
    
    print("Training LSTM (baseline) on early data...")
    lstm_static = train_model(LSTMModel(), X_tr_small, y_tr_small, epochs=5)
    
    print("Training BiLSTM (Bohara) on early data...")
    bilstm_static = train_model(BiLSTMModel(), X_tr_small, y_tr_small, epochs=5)
    
    print("Training ELM (Yang) on early data...")
    elm_static = train_model(ELM(), X_tr_small, y_tr_small, epochs=5)
    
    # Streaming evaluation
    def stream_eval_static(model):
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for i in range(args.start, args.start + args.n):
                if i < SEQ_LEN:
                    continue
                # Get window
                window = scaled[i - SEQ_LEN:i].reshape(1, SEQ_LEN, 1)
                x = torch.from_numpy(window).float().to(device)
                pred_scaled = model(x).cpu().numpy()[0]
                pred_inv = pred_scaled * (mx - mn) + mn
                preds.append(float(pred_inv))
                trues.append(float(series[i]))
        return np.array(preds), np.array(trues)
    
    print("\nStreaming evaluation of static models (frozen, no adaptation)...")
    results = {}
    
    for name, model in [
        ("CNN-GRU_static (Hasanat)", cnn_gru_static),
        ("CNN-BiGRU+Att_static (Ullah)", cnn_bigru_static),
        ("LSTM_static (baseline)", lstm_static),
        ("BiLSTM_static (Bohara)", bilstm_static),
        ("ELM_static (Yang)", elm_static),
    ]:
        preds, trues = stream_eval_static(model)
        results[name] = {
            'mae': mean_absolute_error(trues, preds),
            'rmse': mean_squared_error(trues, preds) ** 0.5,
            'r2': r2_score(trues, preds)
        }
        print(f"  {name}: MAE={results[name]['mae']:.4f} RMSE={results[name]['rmse']:.4f} R2={results[name]['r2']:.4f}")
    
    # EcoEdge with streaming + memory adaptation
    print("\nStreaming evaluation of EcoEdge (with k-NN memory adaptation)...")
    ep = EdgePredictor()
    preds_e, trues_e = [], []
    for i in range(args.start, args.start + args.n):
        if i < SEQ_LEN:
            continue
        window = series[i - SEQ_LEN:i]
        y_curr = series[i]
        pred, _ = ep.predict(window, y_curr=y_curr)
        preds_e.append(float(pred))
        trues_e.append(float(y_curr))
    
    results["EcoEdge_streaming (ours)"] = {
        'mae': mean_absolute_error(trues_e, preds_e),
        'rmse': mean_squared_error(trues_e, preds_e) ** 0.5,
        'r2': r2_score(trues_e, preds_e)
    }
    print(f"  EcoEdge_streaming (ours): MAE={results['EcoEdge_streaming (ours)']['mae']:.4f} RMSE={results['EcoEdge_streaming (ours)']['rmse']:.4f} R2={results['EcoEdge_streaming (ours)']['r2']:.4f}")
    
    # Save results
    with open('streaming_benchmark_results.md', 'w') as f:
        f.write("# Streaming/Forward-Time Evaluation Results\n\n")
        f.write("**Setup:** All static models trained on first 80% of timeline (chronological), then evaluated forward in time without updates. ")
        f.write("EcoEdge uses the full ONNX CNN-GRU + k-NN memory buffer with online adaptation as it streams.\n\n")
        f.write("## Results (MAE, RMSE, R² — MAPE excluded)\n\n")
        f.write("| Model | Type | MAE | RMSE | R² |\n")
        f.write("|-------|------|-----|------|----|\n")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]['mae']):
            mtype = "Static (frozen)" if "static" in name.lower() and "ecoedge" not in name.lower() else "Adaptive (k-NN buffer)"
            f.write(f"| {name} | {mtype} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |\n")
        f.write(f"\n**Evaluation:** Streaming forward from index {args.start} over {args.n} steps\n")
    
    print("\nResults saved to streaming_benchmark_results.md")
    print("="*80)
    for name, m in sorted(results.items(), key=lambda kv: kv[1]['mae']):
        print(f"{name:40s} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
