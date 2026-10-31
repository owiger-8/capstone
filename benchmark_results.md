# Literature Model Benchmark Results

All models evaluated on OPSD household data (single-phase grid import)
on the same train/test split (80/20), sequence length = 60.

## Results (MAE, RMSE, R² — MAPE excluded as per request)

| Model | Paper | MAE | RMSE | R² |
|-------|-------|-----|------|----|
| ELM | Yang2024 | 3.5796 | 3.7085 | 0.9998 |
| EcoEdge | ours | 11.5948 | 12.1879 | 0.9399 |
| CNN-BiGRU+Att | Ullah2025 | 17.5804 | 20.8482 | 0.9944 |
| CNN-GRU | Hasanat2024 | 20.3904 | 23.6518 | 0.9927 |
| LSTM | baseline | 262.0127 | 303.0580 | -0.1913 |
| BiLSTM | Bohara2023 | 265.1929 | 304.2610 | -0.2008 |

**Note:** EcoEdge uses the frozen ONNX CNN-GRU extractor + k-NN memory buffer as described in project_readme.md. Our model's reported metrics from metrics_report.md: MAE=100.41, RMSE=153.27, R²=0.7516.
