"""
Schnelltest der Datenpipeline: prüft, ob Dataset-Klassen korrekt laden,
Bilder die erwartete Shape haben, und zeigt ein Original/Rauschen-Paar an.

Ausführen mit:
    python test_pipeline.py
"""

import matplotlib.pyplot as plt

from config import TRAIN_DIR, IMAGE_SIZE, CHANNELS, GAN_TARGET_CLASS
from src.dataset import BrainMRIDataset, BrainMRIDenoisingDataset


def main() -> None:
    # --- GAN-Dataset testen ---
    gan_ds = BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=GAN_TARGET_CLASS)
    print(f"GAN-Dataset ({GAN_TARGET_CLASS}): {len(gan_ds)} Bilder")
    print(f"  Shape eines Bildes: {gan_ds[0].shape}")
    print(f"  Wertebereich: [{gan_ds[0].min():.3f}, {gan_ds[0].max():.3f}]")

    # --- VAE-Dataset testen ---
    vae_ds = BrainMRIDenoisingDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS)
    noisy, clean = vae_ds[0]
    print(f"\nVAE-Dataset (alle Klassen): {len(vae_ds)} Paare")
    print(f"  Shape: {noisy.shape}")

    # --- Visueller Vergleich Original vs. verrauscht ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(clean.squeeze(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy.squeeze(), cmap="gray")
    axes[1].set_title("Verrauscht (Degradation)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/pipeline_test.png")
    print("\nVergleichsbild gespeichert unter: outputs/pipeline_test.png")
    plt.show()


if __name__ == "__main__":
    main()