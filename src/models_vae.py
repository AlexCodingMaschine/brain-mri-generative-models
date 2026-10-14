"""
Variational Autoencoder (VAE) für 64x64 Graustufen-MRT-Bilder.

Verwendung im Projekt: Denoising. Input = verrauschtes Bild, Target = sauberes Original.

Encoder:  Bild (1x64x64) -> mu und logvar (je latent_dim)
Sampling: z = mu + sigma * eps   (Reparametrisierungs-Trick)
Decoder:  z -> rekonstruiertes Bild (1x64x64), Werte in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, channels: int = 1, feature_maps: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        fm = feature_maps

        # ---------------- Encoder: 64 -> 32 -> 16 -> 8 -> 4 ----------------
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, fm, 4, 2, 1),          # -> (fm, 32, 32)
            nn.BatchNorm2d(fm),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fm, fm * 2, 4, 2, 1),            # -> (fm*2, 16, 16)
            nn.BatchNorm2d(fm * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fm * 2, fm * 4, 4, 2, 1),        # -> (fm*4, 8, 8)
            nn.BatchNorm2d(fm * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fm * 4, fm * 8, 4, 2, 1),        # -> (fm*8, 4, 4)
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = fm * 8 * 4 * 4                  # bei fm=32: 4096

        # Zwei "Köpfe": Mittelwert und log-Varianz der latenten Verteilung
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # ---------------- Decoder: 4 -> 8 -> 16 -> 32 -> 64 ----------------
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1),   # -> (fm*4, 8, 8)
            nn.BatchNorm2d(fm * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1),   # -> (fm*2, 16, 16)
            nn.BatchNorm2d(fm * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1),       # -> (fm, 32, 32)
            nn.BatchNorm2d(fm),
            nn.ReLU(True),

            nn.ConvTranspose2d(fm, channels, 4, 2, 1),     # -> (channels, 64, 64)
            nn.Sigmoid(),                                   # Output in [0, 1]
        )
        self._fm8 = fm * 8

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I). Macht das Sampling differenzierbar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(z.size(0), self._fm8, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1.0):
    """
    Gesamtloss = Rekonstruktionsfehler + kl_weight * KL-Divergenz (jeweils pro Bild gemittelt).

    - Rekonstruktion: MSE (Summe über Pixel). Vergleicht Output mit dem SAUBEREN Original.
    - KL: zieht die latente Verteilung Richtung N(0, I) und ordnet so den latenten Raum.
    """
    batch_size = recon.size(0)
    recon_loss = F.mse_loss(recon, target, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kl_weight * kl, recon_loss, kl
