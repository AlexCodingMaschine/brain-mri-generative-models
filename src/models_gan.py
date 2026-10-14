"""
DCGAN-Architektur (Deep Convolutional GAN) für 64x64 Graustufen-MRT-Bilder.

Generator:     latenter Vektor (z.B. 100-dim Rauschen) -> 64x64x1 Bild
Discriminator: 64x64x1 Bild -> Wahrscheinlichkeit "echt" vs. "gefälscht"

Architektur folgt dem klassischen DCGAN-Paper (Radford et al., 2015):
- Generator: ConvTranspose2d-Schichten, BatchNorm, ReLU (letzte Schicht: Sigmoid,
  da unsere Bilder auf [0,1] normalisiert sind statt auf [-1,1] wie im Original-Paper)
- Discriminator: Conv2d-Schichten, BatchNorm, LeakyReLU
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (latent_dim, 1, 1) -> (feature_maps*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # -> (feature_maps*4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # -> (feature_maps*2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # -> (feature_maps, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # -> (channels, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),  # Output in [0, 1], passend zu unserer Datenpipeline
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z kommt als (batch, latent_dim) -> reshape zu (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (channels, 64, 64) -> (feature_maps, 32, 32)
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (feature_maps*2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (feature_maps*4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (feature_maps*8, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (1, 1, 1)
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img).view(-1, 1).squeeze(1)


def weights_init(m: nn.Module) -> None:
    """DCGAN-übliche Gewichtsinitialisierung (empfohlen im Original-Paper für stabileres Training)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
