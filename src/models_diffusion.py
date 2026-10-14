"""
Diffusion Model für 64x64 Graustufen-MRT-Bilder (vereinfachtes DDPM).

Zwei Teile:
1) GaussianDiffusion: der Zeitplan (wie viel Rauschen bei Schritt t), plus
   q_sample (Vorwärtsprozess: Bild verrauschen) und p_sample (ein Rückwärts-
   schritt: etwas Rauschen entfernen).
2) UNet: das Netzwerk, das bei einem gegebenen verrauschten Bild UND dem
   Zeitschritt t vorhersagt, welches Rauschen hinzugefügt wurde.

Kernidee: Statt direkt ein Bild zu erzeugen, lernt das Netz nur die winzige
Teilaufgabe "sag mir das Rauschen in diesem Bild bei Rauschstufe t".
Erzeugung = viele kleine Rückwärtsschritte hintereinander, startend bei
reinem Rauschen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================================================================
# Zeitplan / Diffusionsprozess
# =======================================================================
class GaussianDiffusion:
    def __init__(self, timesteps: int = 200, device: str = "cpu"):
        self.T = timesteps
        # linearer Beta-Zeitplan wie im DDPM-Originalpaper (Ho et al. 2020),
        # nur mit weniger Schritten für schnelleres Training auf CPU
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Vorwärtsprozess in EINEM Schritt (mathematischer Kurzweg, kein Loop nötig):
        x_t = sqrt(alphabar_t) * x0 + sqrt(1 - alphabar_t) * noise
        """
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """EIN Rückwärtsschritt: x_t -> x_{t-1}, mit vom Netz vorhergesagtem Rauschen."""
        t_batch = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, t_batch)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alphabar_t = self.alphas_cumprod[t]

        # Mittelwert der Rückwärtsverteilung (DDPM-Formel)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alphabar_t)) * pred_noise
        )
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, device: str) -> torch.Tensor:
        """Kompletter Erzeugungsprozess: reines Rauschen -> T Rückwärtsschritte -> Bild."""
        model.eval()
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(0.0, 1.0)


# =======================================================================
# Zeitschritt-Embedding (sagt dem Netz, "wie verrauscht" das Bild gerade ist)
# =======================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Wandelt eine einzelne Zahl t in einen Vektor um -- gleiches Prinzip
    wie Positional Encoding bei Transformern."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    """Conv-Block, der zusätzlich den Zeitschritt-Vektor einspeist."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)  # Zeit-Info einspeisen
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# =======================================================================
# Kleines U-Net
# =======================================================================
class UNet(nn.Module):
    def __init__(self, channels: int = 1, base: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---- Encoder (Downsampling): 64 -> 32 -> 16 ----
        self.in_conv = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim)
        self.pool1 = nn.Conv2d(base, base, 4, stride=2, padding=1)          # 64 -> 32

        self.down2 = ResBlock(base, base * 2, time_dim)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)  # 32 -> 16

        # ---- Bottleneck ----
        self.bottleneck = ResBlock(base * 2, base * 2, time_dim)

        # ---- Decoder (Upsampling): 16 -> 32 -> 64 ----
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)  # 16 -> 32
        self.dec2 = ResBlock(base * 2 + base * 2, base, time_dim)   # + Skip-Connection von down2

        self.up1 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)          # 32 -> 64
        self.dec1 = ResBlock(base + base, base, time_dim)           # + Skip-Connection von down1

        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)          # 64x64, base Kanäle   (Skip 1)
        p1 = self.pool1(d1)                 # 32x32

        d2 = self.down2(p1, t_emb)          # 32x32, base*2 Kanäle (Skip 2)
        p2 = self.pool2(d2)                 # 16x16

        b = self.bottleneck(p2, t_emb)      # 16x16

        u2 = self.up2(b)                    # 32x32
        u2 = self.dec2(torch.cat([u2, d2], dim=1), t_emb)   # Skip-Connection

        u1 = self.up1(u2)                   # 64x64
        u1 = self.dec1(torch.cat([u1, d1], dim=1), t_emb)   # Skip-Connection

        return self.out_conv(u1)            # vorhergesagtes Rauschen, Shape wie Input
