"""
Zeigt die praktische Konsequenz von "geordnet vs. ungeordnet" direkt an den
beiden bereits trainierten VAE-Varianten:

  v1_baseline  (latent=128, kl_weight=1.0)   -> Raum stark geordnet
  kl_0.01      (latent=128, kl_weight=0.01)  -> Raum kaum geordnet

Zwei Experimente, für beide Modelle:

1) ZUFALLSBILDER: z wird aus N(0, I) gewürfelt (kein Bezug zu einem echten
   Bild) und durch den Decoder geschickt. Erwartung: v1_baseline liefert
   plausiblere Bilder, kl_0.01 eher Matsch/Artefakte.

2) INTERPOLATION: zwei echte Val-Bilder werden vom Encoder in z1 und z2
   codiert (deterministisch, über mu). Dazwischen wird linear interpoliert
   und jeder Zwischenpunkt decodiert. Erwartung: v1_baseline zeigt einen
   glatten Übergang, kl_0.01 eher abrupte Sprünge oder unplausible Mischungen.

Ausführen mit:
    python compare_latent_space.py
"""

import torch
from torchvision.utils import save_image

from config import TRAIN_DIR, IMAGE_SIZE, CHANNELS, SEED, DEVICE, OUTPUT_DIR
from src.dataset import BrainMRIDenoisingDataset
from src.models_vae import VAE

# Muss zu den Configs passen, mit denen die Modelle trainiert wurden
MODELS = {
    "v1_baseline (KL=1.0, geordnet)": dict(path=f"{OUTPUT_DIR}/vae_v1_baseline.pt", latent_dim=128, tag="v1_baseline"),
    "kl_0.01 (KL=0.01, ungeordnet)": dict(path=f"{OUTPUT_DIR}/vae_kl_0.01.pt", latent_dim=128, tag="kl_0.01"),
}

N_RANDOM = 8       # wie viele Zufallsbilder pro Modell
N_STEPS = 8        # wie viele Schritte bei der Interpolation (inkl. Start/Ziel)


def load_model(path: str, latent_dim: int) -> VAE:
    model = VAE(latent_dim, CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def random_samples(model: VAE, latent_dim: int, n: int) -> torch.Tensor:
    """Würfelt n Punkte aus N(0, I) und decodiert sie. KEIN Bezug zu echten Bildern."""
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=DEVICE)
        return model.decode(z).cpu()


def interpolate(model: VAE, img_a: torch.Tensor, img_b: torch.Tensor, steps: int) -> torch.Tensor:
    """Codiert zwei echte Bilder (deterministisch über mu) und interpoliert linear dazwischen."""
    with torch.no_grad():
        mu_a, _ = model.encode(img_a.unsqueeze(0).to(DEVICE))
        mu_b, _ = model.encode(img_b.unsqueeze(0).to(DEVICE))
        ts = torch.linspace(0, 1, steps, device=DEVICE).view(-1, 1)
        z_path = (1 - ts) * mu_a + ts * mu_b          # z_t = (1-t)*z1 + t*z2
        return model.decode(z_path).cpu()


def main() -> None:
    torch.manual_seed(SEED)

    # Zwei feste, unterschiedliche Val-Bilder als Interpolations-Endpunkte
    val_ds = BrainMRIDenoisingDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS,
                                      noise_type="gaussian", noise_kwargs={"sigma": 0.0})
    _, img_a = val_ds[0]
    _, img_b = val_ds[len(val_ds) // 2]     # bewusst ein Bild von "weiter weg" im Dataset

    for label, cfg in MODELS.items():
        print(f"\n=== {label} ===")
        model = load_model(cfg["path"], cfg["latent_dim"])

        samples = random_samples(model, cfg["latent_dim"], N_RANDOM)
        fname_rand = f"{OUTPUT_DIR}/latentcheck_random_{cfg['tag']}.png"
        save_image(samples, fname_rand, nrow=N_RANDOM)
        print(f"  Zufallsbilder gespeichert: {fname_rand}")

        path_imgs = interpolate(model, img_a, img_b, N_STEPS)
        fname_interp = f"{OUTPUT_DIR}/latentcheck_interp_{cfg['tag']}.png"
        save_image(path_imgs, fname_interp, nrow=N_STEPS)
        print(f"  Interpolation gespeichert: {fname_interp}")

    # Referenz: die beiden echten Endpunktbilder, zum Vergleich mit der Interpolation
    save_image(torch.stack([img_a, img_b]), f"{OUTPUT_DIR}/latentcheck_endpoints.png", nrow=2)
    print(f"\nEndpunktbilder (Original) gespeichert: {OUTPUT_DIR}/latentcheck_endpoints.png")


if __name__ == "__main__":
    main()
