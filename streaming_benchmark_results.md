# Streaming/Forward-Time Evaluation Results

**Setup:** All static models trained on first 80% of timeline (chronological), then evaluated forward in time without updates. EcoEdge uses the full ONNX CNN-GRU + k-NN memory buffer with online adaptation as it streams.

## Results (MAE, RMSE, R² — MAPE excluded)

| Model | Type | MAE | RMSE | R² |
|-------|------|-----|------|----|
| EcoEdge_streaming (ours) | Adaptive (k-NN buffer) | 62.5487 | 62.9177 | -0.5142 |
| ELM_static (Yang) | Static (frozen) | 2451.9709 | 2452.5017 | -2299.6702 |
| CNN-BiGRU+Att_static (Ullah) | Static (frozen) | 2456.4069 | 2456.9390 | -2308.0030 |
| BiLSTM_static (Bohara) | Static (frozen) | 2456.5338 | 2457.0658 | -2308.2413 |
| CNN-GRU_static (Hasanat) | Static (frozen) | 2456.5338 | 2457.0658 | -2308.2413 |
| LSTM_static (baseline) | Static (frozen) | 2456.5338 | 2457.0658 | -2308.2413 |

**Evaluation:** Streaming forward from index 500000 over 20000 steps
