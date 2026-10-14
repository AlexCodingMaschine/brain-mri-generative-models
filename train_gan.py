"""
Trainingsskript für das GAN (primäres Modell laut Aufgabenstellung).

Trainiert Generator + Discriminator im klassischen adversarialen Wechsel.
Speichert alle paar Epochen Beispielbilder und am Ende die Modellgewichte.

Ausführen mit:
    python train_gan.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import (
    TRAIN_DIR, IMAGE_SIZE, CHANNELS, GAN_TARGET_CLASS,
    BATCH_SIZE, NUM_WORKERS, SEED, DEVICE,
    GAN_LATENT_DIM, GAN_LR, GAN_BETA1, GAN_EPOCHS, OUTPUT_DIR,
)
from src.dataset import BrainMRIDataset
from src.models_gan import Generator, Discriminator, weights_init


def main() -> None:
    torch.manual_seed(SEED)

    # --- Daten ---
    dataset = BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=GAN_TARGET_CLASS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, drop_last=True)
    print(f"Trainiere GAN auf {len(dataset)} Bildern (Klasse: {GAN_TARGET_CLASS})")

    # --- Modelle ---
    generator = Generator(GAN_LATENT_DIM, CHANNELS).to(DEVICE)
    discriminator = Discriminator(CHANNELS).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # --- Loss & Optimizer ---
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))

    # Fester Rausch-Vektor, um den Fortschritt über die Epochen hinweg zu verfolgen
    fixed_noise = torch.randn(16, GAN_LATENT_DIM, device=DEVICE)

    real_label, fake_label = 1.0, 0.0

    for epoch in range(GAN_EPOCHS):
        for i, real_imgs in enumerate(loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # ============================================
            # 1) Discriminator trainieren: max log(D(x)) + log(1 - D(G(z)))
            # ============================================
            discriminator.zero_grad()

            labels = torch.full((batch_size,), real_label, device=DEVICE)
            output = discriminator(real_imgs)
            loss_d_real = criterion(output, labels)
            loss_d_real.backward()

            noise = torch.randn(batch_size, GAN_LATENT_DIM, device=DEVICE)
            fake_imgs = generator(noise)
            labels.fill_(fake_label)
            output = discriminator(fake_imgs.detach())
            loss_d_fake = criterion(output, labels)
            loss_d_fake.backward()

            loss_d = loss_d_real + loss_d_fake
            opt_d.step()

            # ============================================
            # 2) Generator trainieren: max log(D(G(z)))
            # ============================================
            generator.zero_grad()
            labels.fill_(real_label)  # Generator will, dass Discriminator "echt" sagt
            output = discriminator(fake_imgs)
            loss_g = criterion(output, labels)
            loss_g.backward()
            opt_g.step()

            if i % 5 == 0:
                print(f"Epoche [{epoch+1}/{GAN_EPOCHS}] Batch [{i}/{len(loader)}] "
                      f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

        # --- Beispielbilder speichern ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).cpu()
            save_image(samples, f"{OUTPUT_DIR}/gan_samples_epoch_{epoch+1:03d}.png",
                       nrow=4, normalize=False)
            print(f"  -> Beispielbilder gespeichert: gan_samples_epoch_{epoch+1:03d}.png")

    # --- Modelle final speichern ---
    torch.save(generator.state_dict(), f"{OUTPUT_DIR}/generator_final.pt")
    torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator_final.pt")
    print(f"\nTraining abgeschlossen. Modelle gespeichert in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
