"""Compact CNN for 3-class poultry sound classification (healthy / unhealthy / noise)."""
import torch
from torch import nn
from .config import CLASSES


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.15),
    )


class PoultrySoundCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):                 # x: (batch, 1, mels, frames)
        return self.head(self.features(x))
