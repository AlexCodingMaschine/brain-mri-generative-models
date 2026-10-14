"""
Trainingsskript für den VAE (sekundäres Modell): Denoising + Rekonstruktion.

Input:  künstlich verrauschtes MRT-Bild
Target: das saubere Original

Pro Epoche werden Trainings- und Validierungs-Loss sowie der PSNR berechnet.
PSNR (Peak Signal-to-Noise Ratio, in dB): höher = besser. Zum Vergleich wird auch
der PSNR des verrauschten Inputs gegen das Original ausgegeben (= Ausgangslage).

Ausführen mit:
    python train_vae.py
"""

import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from config import (
    TRAIN_DIR, IMAGE_SIZE, CHANNELS, BATCH_SIZE, NUM_WORKERS, SEED, DEVICE,
    VAE_LATENT_DIM, VAE_LR, VAE_EPOCHS, VAE_KL_WEIGHT,
    NOISE_TYPE, GAUSSIAN_NOISE_SIGMA, OUTPUT_DIR,
)
from src.dataset import BrainMRIDenoisingDataset
from src.models_vae import VAE, vae_loss


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mittlerer PSNR über einen Batch (Bildwerte in [0, 1])."""
    mse = ((pred - target) ** 2).flatten(start_dim=1).mean(dim=1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def main() -> None:
    torch.manual_seed(SEED)

    # --- Daten: alle Klassen, 80/20-Split in Train/Val ---
    noise_kwargs = {"sigma": GAUSSIAN_NOISE_SIGMA} if NOISE_TYPE == "gaussian" else {}
    full_ds = BrainMRIDenoisingDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS,
                                       noise_type=NOISE_TYPE, noise_kwargs=noise_kwargs)
    n_val = int(0.2 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    print(f"VAE-Training: {n_train} Train- / {n_val} Val-Bilder | Rauschen: {NOISE_TYPE} {noise_kwargs}")
    print(f"Latente Dimension: {VAE_LATENT_DIM} | KL-Gewicht: {VAE_KL_WEIGHT} | Device: {DEVICE}")

    # --- Modell & Optimizer ---
    model = VAE(VAE_LATENT_DIM, CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)

    # Fester Val-Batch, um den Fortschritt visuell zu verfolgen
    fixed_noisy, fixed_clean = next(iter(val_loader))
    fixed_noisy, fixed_clean = fixed_noisy[:8].to(DEVICE), fixed_clean[:8].to(DEVICE)

    best_val = math.inf
    for epoch in range(VAE_EPOCHS):
        # ---------------- Training ----------------
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            recon, mu, logvar = model(noisy)                  # Input: VERRAUSCHT
            loss, rec, kl = vae_loss(recon, clean, mu, logvar, VAE_KL_WEIGHT)  # Target: SAUBER

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item(); tr_rec += rec.item(); tr_kl += kl.item()
        n_batches = len(train_loader)

        # ---------------- Validierung ----------------
        model.eval()
        val_loss = psnr_in = psnr_out = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                recon, mu, logvar = model(noisy)
                loss, _, _ = vae_loss(recon, clean, mu, logvar, VAE_KL_WEIGHT)
                val_loss += loss.item()
                psnr_in += psnr(noisy, clean)
                psnr_out += psnr(recon, clean)
        n_val_batches = len(val_loader)
        val_loss /= n_val_batches

        print(f"Epoche [{epoch+1}/{VAE_EPOCHS}] "
              f"Train {tr_loss/n_batches:.2f} (Rek {tr_rec/n_batches:.2f} | KL {tr_kl/n_batches:.2f}) "
              f"| Val {val_loss:.2f} "
              f"| PSNR verrauscht {psnr_in/n_val_batches:.2f} dB -> rekonstruiert {psnr_out/n_val_batches:.2f} dB")

        # ---------------- Beispielbilder: Zeile 1 verrauscht, 2 Rekonstruktion, 3 Original ----------------
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                recon_fixed, _, _ = model(fixed_noisy)
            grid = torch.cat([fixed_noisy, recon_fixed, fixed_clean], dim=0).cpu()
            save_image(grid, f"{OUTPUT_DIR}/vae_samples_epoch_{epoch+1:03d}.png", nrow=8)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/vae_best.pt")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/vae_final.pt")
    print(f"\nTraining abgeschlossen. Bester Val-Loss: {best_val:.2f}. Modelle in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
