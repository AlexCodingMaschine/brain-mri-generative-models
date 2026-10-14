"""
Nutzt den trainierten GAN-Generator (Version 2), um synthetische no_tumor-Bilder
zu erzeugen. Diese werden als Tensor-Datei gespeichert, damit train_classifier.py
sie direkt zum Training hinzumischen kann (= Data Augmentation).

Ausführen mit:
    python generate_synthetic.py
"""

import torch
from torchvision.utils import save_image

from config import CHANNELS, GAN_LATENT_DIM, DEVICE, OUTPUT_DIR
from src.models_gan import Generator

N_SYNTHETIC = 430   # 395 echte no_tumor-Bilder + 430 synthetische ≈ 825 (Niveau der anderen Klassen)
GENERATOR_PATH = f"{OUTPUT_DIR}/generator_v2_final.pt"   # Version 2: mehr Vielfalt


def main() -> None:
    generator = Generator(GAN_LATENT_DIM, CHANNELS).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(N_SYNTHETIC, GAN_LATENT_DIM, device=DEVICE)
        images = generator(z).cpu()   # Shape: (N_SYNTHETIC, 1, 64, 64), Werte in [0, 1]

    torch.save(images, f"{OUTPUT_DIR}/synthetic_no_tumor.pt")
    print(f"{N_SYNTHETIC} synthetische no_tumor-Bilder erzeugt und gespeichert "
         f"unter {OUTPUT_DIR}/synthetic_no_tumor.pt")

    # Ein paar Beispiele zur visuellen Kontrolle
    save_image(images[:16], f"{OUTPUT_DIR}/synthetic_no_tumor_preview.png", nrow=4)
    print(f"Vorschau gespeichert: {OUTPUT_DIR}/synthetic_no_tumor_preview.png")


if __name__ == "__main__":
    main()
