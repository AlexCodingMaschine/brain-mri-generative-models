"""
GAN Version 2 -- EINZIGE Änderung gegenüber train_gan.py: Label Smoothing.

Diagnose aus Version 1: Loss_D ~0.16, Loss_G ~5.4 -> der Discriminator wurde
zu sicher (D(G(z)) ~0.005), was tief im flachen Teil der Sigmoid-Kurve liegt
und dem Generator kaum noch brauchbare Gradienten liefert (vanishing gradient).

Gegenmaßnahme: Das Ziel-Label für ECHTE Bilder wird von 1.0 auf 0.9 gesenkt.
Der Discriminator wird dadurch dafür "bestraft", zu übermütig zu werden, und
bleibt eher in einem Bereich der Sigmoid-Kurve mit brauchbarer Steigung.
Das Fake-Label bleibt bei 0.0 (nur einseitiges Smoothing -- Standardempfehlung).

Alles andere (Architektur, Lernrate, Batch-Size, Epochen, Daten) ist
IDENTISCH zu Version 1, damit der Vergleich aussagekräftig bleibt.

Ausführen mit:
    python train_gan_v2.py
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

REAL_LABEL_SMOOTH = 0.9   # <- die einzige inhaltliche Änderung ggü. Version 1 (dort: 1.0)


def main() -> None:
    torch.manual_seed(SEED)

    dataset = BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=GAN_TARGET_CLASS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=True)
    print(f"GAN v2 (Label Smoothing={REAL_LABEL_SMOOTH}) auf {len(dataset)} Bildern (Klasse: {GAN_TARGET_CLASS})")

    generator = Generator(GAN_LATENT_DIM, CHANNELS).to(DEVICE)
    discriminator = Discriminator(CHANNELS).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))

    fixed_noise = torch.randn(16, GAN_LATENT_DIM, device=DEVICE)
    fake_label = 0.0

    for epoch in range(GAN_EPOCHS):
        for i, real_imgs in enumerate(loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # ---- Discriminator ----
            discriminator.zero_grad()

            # NEU: Ziel-Label für echte Bilder ist 0.9 statt 1.0 (Label Smoothing)
            labels = torch.full((batch_size,), REAL_LABEL_SMOOTH, device=DEVICE)
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

            # ---- Generator ----
            generator.zero_grad()
            labels.fill_(1.0)  # Generator-Ziel bleibt bei 1.0 -- er soll "maximal echt" wirken
            output = discriminator(fake_imgs)
            loss_g = criterion(output, labels)
            loss_g.backward()
            opt_g.step()

            if i % 5 == 0:
                print(f"Epoche [{epoch+1}/{GAN_EPOCHS}] Batch [{i}/{len(loader)}] "
                      f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).cpu()
            save_image(samples, f"{OUTPUT_DIR}/gan_v2_samples_epoch_{epoch+1:03d}.png",
                      nrow=4, normalize=False)
            print(f"  -> Beispielbilder gespeichert: gan_v2_samples_epoch_{epoch+1:03d}.png")

    torch.save(generator.state_dict(), f"{OUTPUT_DIR}/generator_v2_final.pt")
    torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator_v2_final.pt")
    print(f"\nTraining abgeschlossen. Modelle gespeichert in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
