import torch
import torch.nn as nn


class CNNGRUFeatureExtractor(nn.Module):
    def __init__(self, seq_len=60, num_features=1, latent_dim=32):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=16, hidden_size=latent_dim, batch_first=True)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        features = self.dropout(self.relu(self.conv1d(x_perm))).permute(0, 2, 1)
        gru_out, _ = self.gru(features)
        latent_vector = self.dropout(gru_out[:, -1, :])
        prediction = self.fc(latent_vector)
        return prediction, latent_vector
