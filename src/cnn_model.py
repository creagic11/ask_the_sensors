"""
1D-CNN Baseline and Quantized Edge Backbone.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Features:
- Compact 1D-CNN designed for 25 Hz 6-channel IMU windows (64 samples).
- Supports FP32 and INT8 Dynamic Quantization for edge Pareto benchmark.
- Fast training and lightweight inference footprint (<1MB on disk).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict

class TinyHAR1DCNN(nn.Module):
    """
    Compact 1D-CNN backbone for 7-class Human Activity Recognition.
    Input: (batch_size, 6, 64) -> Output: (batch_size, 7)
    """
    def __init__(self, num_channels: int = 6, num_classes: int = 7):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 -> 32
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 32 -> 16
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (batch, 64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, window_len, channels) or (batch, channels, window_len)
        if x.dim() == 3 and x.size(1) != 6 and x.size(2) == 6:
            x = x.permute(0, 2, 1)
        feats = self.conv_block(x)
        logits = self.classifier(feats)
        return logits

def get_quantized_cnn(model: TinyHAR1DCNN) -> nn.Module:
    """Applies dynamic INT8 quantization to linear layers for edge deployment."""
    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def train_cnn(
    model: TinyHAR1DCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3
) -> Dict[str, float]:
    """Trains the 1D-CNN backbone on windowed sensor data."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to PyTorch tensors (N, C, L)
    tensor_x = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    tensor_y = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)

    model.eval()
    return {"final_loss": epoch_loss / len(X_train)}
