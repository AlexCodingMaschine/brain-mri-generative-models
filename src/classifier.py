"""
Kleiner CNN-Klassifikator für die 4 Klassen (glioma, meningioma, pituitary, no_tumor).
Dient hier NICHT als Hauptmodell des Projekts, sondern als einfaches "Messgerät",
um zu testen, ob synthetische GAN-Bilder als Data Augmentation nützlich sind.
"""

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),          # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),        # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
