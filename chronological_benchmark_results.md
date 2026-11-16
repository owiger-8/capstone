# Chronological Evaluation Results

**Setup:** Time-series split (chronological, no shuffle). Train on first 80%, test on last 20%. Static models are frozen. EcoEdge uses online k-NN memory adaptation.

## Results (MAE, RMSE, R² — MAPE excluded)

| Model | Type | MAE | RMSE | R² |
|-------|------|-----|------|----|
| CNN-GRU (Hasanat2024) | Static (frozen) | 0.0000 | 0.0000 | 1.0000 |
| CNN-BiGRU+Att (Ullah2025) | Static (frozen) | 0.0000 | 0.0000 | 1.0000 |
| LSTM (baseline) | Static (frozen) | 0.0000 | 0.0000 | 1.0000 |
| BiLSTM (Bohara2023) | Static (frozen) | 0.0000 | 0.0000 | 1.0000 |
| ELM (Yang2024) | Static (frozen) | 0.0000 | 0.0000 | 1.0000 |
| EcoEdge (ours, adaptive) | Adaptive (k-NN buffer) | 27.7748 | 27.7779 | 0.0000 |
