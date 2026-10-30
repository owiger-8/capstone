# Real-Time Household Carbon Emission & Energy Demand Forecasting

## Table of Contents
1. [About](#about)
2. [Introduction](#introduction)
3. [Use Case](#use-case)
4. [Device & Specification](#device--specification)
5. [Architecture](#architecture)
6. [Dataset](#dataset)
7. [Training](#training)
8. [Testing & Evaluation](#testing--evaluation)
9. [Metrics](#metrics)
10. [Deployment](#deployment)

---

## About
This project implements an adaptive, edge-deployed Machine Learning pipeline to predict real-time carbon emissions and electricity demand for individual households. By utilizing a hybrid **1D-CNN + GRU** (Convolutional Neural Network + Gated Recurrent Unit) architecture combined with a **Non-Parametric Latent Memory Buffer (k-NN)**, the system instantly adapts to new homes, appliances, and schedules without requiring heavy on-device model retraining.

## Introduction
Time-series forecasting for household energy is difficult due to **Domain Shift** (Concept Drift). A neural network trained on Household A typically fails on Household B because of different base loads, occupant behaviors, and appliance footprints (e.g., EVs, solar panels, heat pumps).

Traditional Reinforcement Learning or continuous backpropagation on edge devices consumes too much memory and degrades flash storage. This project introduces **Strategy 3: Frozen Base Extractor with Online Memory**. The CNN-GRU extracts generalized features, while a dynamic, local vector database learns the unique footprint of the deployed household in real-time.

## Use Case
* **Smart Grid Demand Response:** Predicting peak loads to shift appliance usage.
* **Home Energy Management Systems (HEMS):** Optimizing solar battery charge/discharge cycles.
* **Carbon Tracking:** Providing real-time estimates of household carbon footprints to encourage sustainable behaviors.

## Device & Specification
This system is strictly optimized for resource-constrained edge hardware.
* **Target Device:** Raspberry Pi 5
* **CPU:** Broadcom BCM2712, 2.4GHz quad-core 64-bit Arm Cortex-A76
* **RAM:** 2GB LPDDR4X (System requires <150MB total runtime memory)
* **Storage:** 128GB High-Endurance MicroSD Card
* **Runtime Environments:** `onnxruntime` (CPU), `hnswlib` (Vector Search), `numpy`. (Raw PyTorch is avoided on the edge to prevent Out-Of-Memory errors).

## Architecture
The system operates in two phases: Offline Pre-training and Online Edge Adaptation.

```text
[ Live 1-Min Sensor Stream ] ---> [ 60-Step Input Buffer ]
                                          |
                                          v
                         +---------------------------------+
                         |  FROZEN 1D-CNN + GRU EXTRACTOR  |  <-- ONNX Engine
                         +---------------------------------+
                                /                   \
        Base Forecast (y_base) /                     \ Latent Vector (h_t)
                              /                       v
                             |             +---------------------+
                             |             | LOCAL MEMORY INDEX  | <-- HNSWLIB
                             |             | (Dynamic RAM Buffer)|
                             |             +---------------------+
                             |                        |
                             |                        v (Query Top-k Matches)
                             |             Historical Target Match (y_mem)
                              \                       /
                               v                     v
                    +----------------------------------+
                    |   ADAPTIVE BLENDING & OUTPUT     |
                    |  y_hat = 0.6(y_b) + 0.4(y_mem)   |
                    +----------------------------------+
```

1. **1D-CNN:** Extracts local spatial features (spikes in load).
2. **GRU:** Captures the temporal sequence of the past 60 minutes.
3. **Latent Memory:** Maps the GRU's final hidden state into a 32-dimensional embedding, storing it alongside the actual energy demand to build a personalized profile over time.

## Dataset
This repository uses the **Open Power System Data (OPSD) Household Dataset**.
* **Source:** [OPSD Household Data](https://data.open-power-system-data.org/household_data/)
* **Resolution:** 1-minute interval time series.
* **Target Variables:** `DE_KN_residential1_grid_import`, etc.
* **Preprocessing:** Rolling window creation (sequence length = 60) and Min-Max scaling.

## Training
The offline training phase occurs on a standard PC/Server using PyTorch.

```python
import torch
import torch.nn as nn

class CNNGRUFeatureExtractor(nn.Module):
    def __init__(self, seq_len=60, num_features=1, latent_dim=32):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=16, hidden_size=latent_dim, batch_first=True)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        features = self.relu(self.conv1d(x_perm)).permute(0, 2, 1)
        gru_out, _ = self.gru(features)
        latent_vector = gru_out[:, -1, :] 
        prediction = self.fc(latent_vector)
        return prediction, latent_vector

# Save checkpoint & Export to ONNX after training
# torch.onnx.export(model, dummy_input, "cnn_gru_base.onnx", ...)
```

## Testing & Evaluation
We benchmark the hybrid model against standard baselines using a simulated streaming environment to test real-time adaptation.

```python
# Pseudo-code for evaluation loop
for x_curr, y_curr in test_stream:
    base_pred, latent_h = onnx_session.run(None, {input_name: x_curr})
    
    if memory_index.count >= 10:
        neighbors = memory_index.query(latent_h, k=5)
        memory_pred = np.mean(neighbors.values)
        final_pred = 0.6 * base_pred + 0.4 * memory_pred
    else:
        final_pred = base_pred
        
    memory_index.add(latent_h, y_curr)
```

## Metrics
The project evaluates real-time predictions using the following standard metrics:
1. **MAE (Mean Absolute Error):** Measures the average magnitude of forecasting errors in kilowatts.
2. **RMSE (Root Mean Squared Error):** Heavily penalizes large forecasting errors (e.g., missing a sudden EV charging spike).
3. **MAPE (Mean Absolute Percentage Error):** Evaluates relative error percentage.
4. **R² Score:** Indicates how well the model captures the overall variance of the household's energy footprint.

## Deployment
Deployment on the Raspberry Pi 5 requires strict memory and storage management.

### 1. Edge Setup
```bash
sudo apt update && sudo apt install -y python3-pip python3-numpy
pip3 install onnxruntime hnswlib pandas scikit-learn
```

### 2. Execution Strategy (Protecting the SD Card)
To prevent wearing out the 128GB MicroSD card with constant 1-minute writes:
* **RAM Buffer:** The `hnswlib` index and recent logs are held entirely in the 2GB RAM.
* **Batch Flushing:** A background thread writes the updated vector index to the SD card once every 6 hours rather than every minute.
* **Lightweight Inference:** By relying on `onnxruntime` (CPU execution provider) instead of raw PyTorch, the total runtime footprint is kept safely under 120MB, leaving plenty of overhead for the OS.