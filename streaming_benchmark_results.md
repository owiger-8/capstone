# Streaming/Forward-Time Evaluation Results

**Setup:** All static models trained on first 80% of timeline (chronological), then evaluated forward in time without updates. EcoEdge uses the full ONNX CNN-GRU + k-NN memory buffer with online adaptation as it streams.

## Results (MAE, RMSE, R² — MAPE excluded)

| Model | Type | MAE | RMSE | R² |
|-------|------|-----|------|----|
| EcoEdge_streaming (ours) | Adaptive (k-NN buffer) | 60.3973 | 74.8898 | 0.9717 |
| ELM_static (Yang) | Static (frozen) | 600.7502 | 747.3726 | -1.8162 |
| CNN-BiGRU+Att_static (Ullah) | Static (frozen) | 601.4990 | 748.3949 | -1.8239 |
| BiLSTM_static (Bohara) | Static (frozen) | 601.5856 | 748.4968 | -1.8246 |
| CNN-GRU_static (Hasanat) | Static (frozen) | 601.5856 | 748.4968 | -1.8246 |
| LSTM_static (baseline) | Static (frozen) | 601.5856 | 748.4968 | -1.8246 |

**Evaluation:** Streaming forward from index 200000 over 200000 steps
