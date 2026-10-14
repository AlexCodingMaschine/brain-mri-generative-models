"""
Trainingsskript für das Diffusion Model.

Trainingsaufgabe pro Batch:
1. Zufälligen Zeitschritt t für jedes Bild ziehen
2. Zufälliges Rauschen ziehen
3. Das Bild mit genau diesem Rauschen bei genau diesem t verrauschen (q_sample)
4. Das Netz soll aus dem verrauschten Bild + t das Rauschen zurückvorhersagen
5. Loss = wie weit die Vorhersage vom tatsächlichen Rauschen abweicht (MSE)

Kein Discriminator, kein Encoder -- nur EIN Netz mit einem festen, einfachen Ziel.
Deshalb ist Diffusion-Training in der Regel deutlich stabiler als GAN-Training.

Ausführen mit:
    python train_diffusion.py
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import (
    TRAIN_DIR, IMAGE_SIZE, CHANNELS, DIFFUSION_TARGET_CLASS,
    BATCH_SIZE, NUM_WORKERS, SEED, DEVICE, OUTPUT_DIR,
    DIFFUSION_TIMESTEPS, DIFFUSION_BASE_CHANNELS, DIFFUSION_LR, DIFFUSION_EPOCHS,
)
from src.dataset import BrainMRIDataset
from src.models_diffusion import UNet, GaussianDiffusion

# Eigene Dateinamen je nach Klassen-Einstellung, damit ein Lauf mit allen Klassen
# nicht versehentlich den Checkpoint eines Laufs mit nur einer Klasse überschreibt
# oder davon fortsetzt.
TAG = DIFFUSION_TARGET_CLASS if DIFFUSION_TARGET_CLASS else "all_classes"


def main() -> None:
    torch.manual_seed(SEED)

    dataset = BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=DIFFUSION_TARGET_CLASS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=True)
    print(f"Diffusion-Training auf {len(dataset)} Bildern (Klasse: {DIFFUSION_TARGET_CLASS or 'ALLE'})")
    print(f"Zeitschritte T={DIFFUSION_TIMESTEPS} | Device: {DEVICE} | Tag: {TAG}")

    diffusion = GaussianDiffusion(DIFFUSION_TIMESTEPS, device=DEVICE)
    model = UNet(CHANNELS, DIFFUSION_BASE_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=DIFFUSION_LR)

    # ---- Fortsetzen, falls schon ein Checkpoint MIT DEMSELBEN TAG existiert ----
    checkpoint_path = f"{OUTPUT_DIR}/diffusion_checkpoint_{TAG}.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"Setze Training fort ab Epoche {start_epoch + 1} (Checkpoint gefunden)")

    for epoch in range(start_epoch, DIFFUSION_EPOCHS):
        model.train()
        total_loss = 0.0
        for x0 in loader:
            x0 = x0.to(DEVICE)
            batch_size = x0.size(0)

            # Schritt 1+2: zufälliger Zeitpunkt und zufälliges Rauschen pro Bild
            t = torch.randint(0, DIFFUSION_TIMESTEPS, (batch_size,), device=DEVICE)
            noise = torch.randn_like(x0)

            # Schritt 3: Bild bei Zeitpunkt t verrauschen
            x_t = diffusion.q_sample(x0, t, noise)

            # Schritt 4: Netz soll das Rauschen erraten
            pred_noise = model(x_t, t)

            # Schritt 5: Fehler zwischen echtem und vorhergesagtem Rauschen
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoche [{epoch+1}/{DIFFUSION_EPOCHS}] Loss (Rausch-Vorhersage-Fehler): {avg_loss:.4f}")

        # Alle 5 Epochen: kompletten Erzeugungsprozess einmal durchlaufen lassen
        # UND einen Checkpoint speichern (falls das Training abbricht, geht nichts verloren)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = diffusion.sample(model, (8, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), DEVICE)
            save_image(samples, f"{OUTPUT_DIR}/diffusion_samples_{TAG}_epoch_{epoch+1:03d}.png", nrow=8)
            print(f"  -> Beispielbilder gespeichert (Erzeugung aus reinem Rauschen, {DIFFUSION_TIMESTEPS} Schritte)")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"  -> Checkpoint gespeichert (Epoche {epoch+1})")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/diffusion_final_{TAG}.pt")
    print(f"\nTraining abgeschlossen. Modell gespeichert in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()