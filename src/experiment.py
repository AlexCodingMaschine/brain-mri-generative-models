"""
Bausteine für die VAE-Experimente:
- feste (reproduzierbare) verrauschte Validierungsmenge, damit alle Modelle
  und Baselines auf exakt denselben Bildern verglichen werden
- Training einer VAE-Konfiguration
- Metriken: PSNR und SSIM
- klassische Baselines: Gauß-Filter und Median-Filter
"""

import time

import numpy as np
import torch
import torch.optim as optim
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from config import (
    TRAIN_DIR, IMAGE_SIZE, CHANNELS, BATCH_SIZE, NUM_WORKERS, SEED, DEVICE, OUTPUT_DIR,
)
from src.dataset import BrainMRIDenoisingDataset
from src.models_vae import VAE, vae_loss


# ---------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------
def psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mittlerer PSNR (dB) über alle Bilder. Werte in [0, 1]."""
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mittlerer SSIM (1.0 = identisch). Misst Struktur, nicht nur Pixelfehler."""
    p = pred.squeeze(1).cpu().numpy()
    t = target.squeeze(1).cpu().numpy()
    return float(np.mean([structural_similarity(a, b, data_range=1.0) for a, b in zip(p, t)]))


# ---------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------
def make_data(noise_type: str, noise_kwargs: dict):
    """Gleicher 80/20-Split in jedem Experiment (fester Seed)."""
    full_ds = BrainMRIDenoisingDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS,
                                       noise_type=noise_type, noise_kwargs=noise_kwargs)
    n_val = int(0.2 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    return train_loader, val_ds


def fixed_eval_set(val_ds):
    """Verrauscht jedes Val-Bild GENAU EINMAL mit festem Seed.
    So sind alle Auswertungen vergleichbar (kein zufälliges Neu-Würfeln des Rauschens)."""
    torch.manual_seed(SEED)
    pairs = [val_ds[i] for i in range(len(val_ds))]
    noisy = torch.stack([p[0] for p in pairs])
    clean = torch.stack([p[1] for p in pairs])
    return noisy, clean


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------
def classical_baselines(noisy: torch.Tensor) -> dict:
    """Klassisches Entrauschen ohne Lernen. Gibt Tensoren (N,1,H,W) zurück."""
    n = noisy.squeeze(1).numpy()
    gauss = np.stack([gaussian_filter(x, sigma=1.0) for x in n])
    median = np.stack([median_filter(x, size=3) for x in n])
    to_t = lambda a: torch.from_numpy(a).float().unsqueeze(1)
    return {"gauss": to_t(gauss), "median": to_t(median)}


# ---------------------------------------------------------------------
# VAE-Auswertung
# ---------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model: VAE, noisy: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Deterministische Rekonstruktion: nutzt den Mittelwert mu statt eines zufälligen z."""
    model.eval()
    outs = []
    for i in range(0, len(noisy), batch_size):
        x = noisy[i:i + batch_size].to(DEVICE)
        mu, _ = model.encode(x)
        outs.append(model.decode(mu).cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------------
# Ein komplettes Experiment
# ---------------------------------------------------------------------
def run_experiment(name: str, latent_dim: int, kl_weight: float, sigma: float,
                   epochs: int, lr: float = 1e-3) -> dict:
    start = time.time()
    torch.manual_seed(SEED)
    noise_kwargs = {"sigma": sigma}

    train_loader, val_ds = make_data("gaussian", noise_kwargs)
    val_noisy, val_clean = fixed_eval_set(val_ds)

    model = VAE(latent_dim, CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n=== {name}: latent={latent_dim}, kl={kl_weight}, sigma={sigma}, epochs={epochs} ===")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            recon, mu, logvar = model(noisy)
            loss, _, _ = vae_loss(recon, clean, mu, logvar, kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            p = psnr_batch(reconstruct(model, val_noisy), val_clean)
            print(f"  Epoche {epoch+1}/{epochs} | Train-Loss {total/len(train_loader):.2f} | Val-PSNR {p:.2f} dB")

    # ---------- Endauswertung auf der festen Val-Menge ----------
    recon = reconstruct(model, val_noisy)
    base = classical_baselines(val_noisy)

    result = {
        "name": name, "latent_dim": latent_dim, "kl_weight": kl_weight,
        "sigma": sigma, "epochs": epochs,
        "psnr_noisy": psnr_batch(val_noisy, val_clean),
        "ssim_noisy": ssim_batch(val_noisy, val_clean),
        "psnr_vae": psnr_batch(recon, val_clean),
        "ssim_vae": ssim_batch(recon, val_clean),
        "psnr_gauss": psnr_batch(base["gauss"], val_clean),
        "ssim_gauss": ssim_batch(base["gauss"], val_clean),
        "psnr_median": psnr_batch(base["median"], val_clean),
        "ssim_median": ssim_batch(base["median"], val_clean),
        "minutes": round((time.time() - start) / 60, 1),
    }
    result = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}

    # Bildraster: verrauscht / VAE / Gauß-Filter / Original (je 8 Beispiele)
    grid = torch.cat([val_noisy[:8], recon[:8], base["gauss"][:8], val_clean[:8]])
    save_image(grid, f"{OUTPUT_DIR}/vae_exp_{name}.png", nrow=8)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/vae_{name}.pt")

    print(f"  -> PSNR verrauscht {result['psnr_noisy']} | VAE {result['psnr_vae']} | "
          f"Gauß {result['psnr_gauss']} | Median {result['psnr_median']}")
    print(f"  -> SSIM  verrauscht {result['ssim_noisy']} | VAE {result['ssim_vae']} | "
          f"Gauß {result['ssim_gauss']} | Median {result['ssim_median']}")
    return result
