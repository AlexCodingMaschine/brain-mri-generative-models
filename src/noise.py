"""
Künstliche Degradation von MRT-Bildern für die VAE-Denoising-Aufgabe.

Die Aufgabenstellung verlangt, kontrolliertes Rauschen selbst einzuführen,
da kein Datensatz mit garantierten (sauber, verrauscht)-Paaren existiert.
Jede Funktion nimmt einen Tensor im Bereich [0, 1] und Shape (C, H, W) an.
"""

import torch
import torch.nn.functional as F


def add_gaussian_noise(img: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """Fügt additives Gaußsches Rauschen hinzu (simuliert z.B. Scanner-Grundrauschen)."""
    noise = torch.randn_like(img) * sigma
    noisy = img + noise
    return torch.clamp(noisy, 0.0, 1.0)


def add_salt_pepper_noise(img: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
    """Fügt Salt-and-Pepper-Rauschen hinzu (simuliert defekte Pixel/Artefakte)."""
    noisy = img.clone()
    mask = torch.rand_like(img)
    noisy[mask < amount / 2] = 0.0       # "Pepper"
    noisy[mask > 1 - amount / 2] = 1.0   # "Salt"
    return noisy


def add_gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """Simuliert Bewegungsunschärfe / geringe Bildschärfe durch Gaußschen Blur."""
    channels = img.shape[0]
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    img_batched = img.unsqueeze(0)  # (1, C, H, W)
    blurred = F.conv2d(img_batched, kernel, padding=kernel_size // 2, groups=channels)
    return blurred.squeeze(0).clamp(0.0, 1.0)


def apply_degradation(img: torch.Tensor, noise_type: str, **kwargs) -> torch.Tensor:
    """Zentrale Dispatch-Funktion basierend auf config.NOISE_TYPE."""
    if noise_type == "gaussian":
        return add_gaussian_noise(img, sigma=kwargs.get("sigma", 0.15))
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(img, amount=kwargs.get("amount", 0.05))
    elif noise_type == "blur":
        return add_gaussian_blur(img, kernel_size=kwargs.get("kernel_size", 5),
                                  sigma=kwargs.get("sigma", 1.5))
    else:
        raise ValueError(f"Unbekannter noise_type: {noise_type}")
