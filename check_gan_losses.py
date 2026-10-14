"""
Misst die typischen GAN-Losses aus den GESPEICHERTEN Modellen nach (falls das
Trainings-Log nicht mehr vorhanden ist).

Verwendet dieselbe Loss-Berechnung wie train_gan.py:
  Loss_D = BCE(D(echt), Label_echt) + BCE(D(fake), 0)
  Loss_G = BCE(D(fake), 1)
und gibt Mittelwert, Minimum und Maximum über mehrere Batches aus, dazu die
mittlere Discriminator-Ausgabe für echte und generierte Bilder.

WICHTIG: Das ist der Zustand am ENDE des Trainings (Epoche 100), nicht der Verlauf.
Einzelne Batches schwanken (siehe Version-2-Log), deshalb Mittelwert und Spanne.

Ausführen im Projektordner (venv aktiviert):
    python check_gan_losses.py            # Version 1: generator_final.pt, Label-Ziel 1.0
    python check_gan_losses.py --v2       # Version 2: generator_v2_final.pt, Label-Ziel 0.9
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR, IMAGE_SIZE, CHANNELS, GAN_TARGET_CLASS,
    BATCH_SIZE, SEED, DEVICE, GAN_LATENT_DIM, OUTPUT_DIR,
)
from src.dataset import BrainMRIDataset
from src.models_gan import Generator, Discriminator

N_REPEATS = 10   # wie oft der (kleine) Datensatz durchlaufen wird -> mehr Batches für stabile Mittelwerte


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="Version 2 (Label Smoothing 0.9) statt Version 1")
    args = ap.parse_args()

    suffix = "v2_" if args.v2 else ""
    real_label = 0.9 if args.v2 else 1.0
    gen_path = f"{OUTPUT_DIR}/generator_{suffix}final.pt"
    disc_path = f"{OUTPUT_DIR}/discriminator_{suffix}final.pt"

    torch.manual_seed(SEED)
    dataset = BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=GAN_TARGET_CLASS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    generator = Generator(GAN_LATENT_DIM, CHANNELS).to(DEVICE)
    discriminator = Discriminator(CHANNELS).to(DEVICE)
    generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
    discriminator.load_state_dict(torch.load(disc_path, map_location=DEVICE))
    # train()-Modus wie im Training (BatchNorm nutzt die Statistik des jeweiligen Batches)
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()
    loss_d, loss_g, d_real, d_fake = [], [], [], []

    with torch.no_grad():
        for _ in range(N_REPEATS):
            for real in loader:
                real = real.to(DEVICE)
                b = real.size(0)

                out_real = discriminator(real)
                l_real = criterion(out_real, torch.full((b,), real_label, device=DEVICE))

                z = torch.randn(b, GAN_LATENT_DIM, device=DEVICE)
                fake = generator(z)
                out_fake = discriminator(fake)
                l_fake = criterion(out_fake, torch.zeros(b, device=DEVICE))
                l_gen = criterion(out_fake, torch.ones(b, device=DEVICE))

                loss_d.append((l_real + l_fake).item())
                loss_g.append(l_gen.item())
                d_real.append(out_real.mean().item())
                d_fake.append(out_fake.mean().item())

    def stats(name, values):
        t = torch.tensor(values)
        print(f"  {name:<10} Mittelwert {t.mean():7.4f} | Min {t.min():7.4f} | Max {t.max():7.4f}")

    print(f"\n=== GAN {'Version 2' if args.v2 else 'Version 1'}: Zustand am Trainingsende "
          f"({len(loss_d)} Batches, Label-Ziel echt = {real_label}) ===")
    stats("Loss_D", loss_d)
    stats("Loss_G", loss_g)
    stats("D(echt)", d_real)
    stats("D(fake)", d_fake)
    print("\nLesehilfe: Ist Loss_D niedrig und Loss_G hoch, dominiert der Discriminator "
          "(D(echt) nahe 1, D(fake) nahe 0).")


if __name__ == "__main__":
    main()
