"""
Quantitative Bewertung von Qualität UND Diversität der GAN-generierten Bilder
mittels FID (Fréchet Inception Distance) -- deckt Aufgabenpunkt 4c ab
("Evaluate quality and diversity using appropriate quantitative ... criteria").

Wie FID funktioniert:
1. Ein vortrainiertes Bildnetz (InceptionV3) wandelt jedes Bild in einen
   2048-dimensionalen Merkmalsvektor um (NICHT die rohen Pixel).
2. Für die echten und die generierten Bilder wird je eine Normalverteilung
   über diese Vektoren geschätzt (Mittelwert + Kovarianz).
3. FID misst den Abstand zwischen diesen beiden Verteilungen.
Niedriger Wert = die generierten Bilder "verhalten sich" im Merkmalsraum
ähnlich wie echte Bilder (Qualität UND Diversität fließen hier zusammen ein).

WICHTIGE EINSCHRÄNKUNG: FID ist für große Stichproben (oft >2000 Bilder)
konzipiert. Mit ~105 echten no_tumor-Testbildern ist der Schätzwert deutlich
unsicherer als in der Literatur üblich. Der Sanity-Check unten (reale Bilder
gegen sich selbst, in zwei Hälften gesplittet) zeigt, wie groß dieses
"Rauschen" bei dieser Stichprobengröße ungefähr ist -- er liefert eine
ungefähre Untergrenze, unterhalb derer ein FID-Unterschied nicht mehr
aussagekräftig interpretiert werden sollte.

Voraussetzung: generator_final.pt (v1) und generator_v2_final.pt (v2) existieren.
Lädt beim ersten Ausführen automatisch vortrainierte InceptionV3-Gewichte aus
dem Internet (einmalig, ca. 100 MB).

Ausführen mit:
    python compute_fid.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

from config import TEST_DIR, IMAGE_SIZE, CHANNELS, GAN_LATENT_DIM, SEED, DEVICE, OUTPUT_DIR
from src.dataset import BrainMRIDataset
from src.models_gan import Generator

N_SAMPLES = 105          # = Anzahl echter no_tumor-Testbilder, für einen fairen Vergleich
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_inception() -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = nn.Identity()   # letzte Klassifikationsschicht entfernen -> Output = 2048-dim Feature
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def get_features(model: nn.Module, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """images: (N, 1, H, W) in [0,1]. Gibt (N, 2048) Feature-Array zurück."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(DEVICE)
        batch = batch.repeat(1, 3, 1, 1)                               # Graustufen -> 3 Kanäle
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - IMAGENET_MEAN.to(DEVICE)) / IMAGENET_STD.to(DEVICE)
        out = model(batch)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    mu1, mu2 = feat1.mean(axis=0), feat2.mean(axis=0)
    sigma1 = np.cov(feat1, rowvar=False)
    sigma2 = np.cov(feat2, rowvar=False)

    diff = mu1 - mu2
    result = linalg.sqrtm(sigma1 @ sigma2)   # je nach scipy-Version Tensor ODER (Tensor, Fehlerschätzung)
    covmean = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(covmean):             # numerische Rundungsfehler -> Imaginärteil verwerfen
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def main() -> None:
    torch.manual_seed(SEED)
    print("Lade InceptionV3 (einmalig ggf. Download)...")
    inception = load_inception()

    # ---- Echte Testbilder (nie fürs Training verwendet) ----
    real_ds = BrainMRIDataset(TEST_DIR, IMAGE_SIZE, CHANNELS, target_class="no_tumor")
    real_imgs = torch.stack([real_ds[i] for i in range(len(real_ds))])
    print(f"Echte Testbilder: {len(real_imgs)}")

    # Sanity-Check: echte Bilder in zwei Hälften gesplittet, FID gegeneinander.
    # Das zeigt das "Rauschlevel" von FID bei dieser Stichprobengröße.
    half = len(real_imgs) // 2
    real_feats_all = get_features(inception, real_imgs)
    fid_sanity = frechet_distance(real_feats_all[:half], real_feats_all[half:])
    print(f"\nSanity-Check (echt vs. echt, gesplittet): FID = {fid_sanity:.1f}")
    print("-> Werte UNTER diesem Niveau sind bei dieser Stichprobengröße kaum noch aussagekräftig.\n")

    results = {}
    for tag, path in [("GAN v1", f"{OUTPUT_DIR}/generator_final.pt"),
                      ("GAN v2", f"{OUTPUT_DIR}/generator_v2_final.pt")]:
        generator = Generator(GAN_LATENT_DIM, CHANNELS).to(DEVICE)
        generator.load_state_dict(torch.load(path, map_location=DEVICE))
        generator.eval()
        with torch.no_grad():
            z = torch.randn(N_SAMPLES, GAN_LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z).cpu()

        fake_feats = get_features(inception, fake_imgs)
        fid = frechet_distance(real_feats_all, fake_feats)
        results[tag] = fid
        print(f"FID ({tag} vs. echte Testbilder): {fid:.1f}")

    print("\n" + "=" * 50)
    print(f"{'Vergleich':<30}{'FID':>15}")
    print("-" * 50)
    print(f"{'echt vs. echt (Sanity-Check)':<30}{fid_sanity:>15.1f}")
    for tag, fid in results.items():
        print(f"{'echt vs. ' + tag:<30}{fid:>15.1f}")
    print("=" * 50)
    print("Niedriger = besser (generierte Bilder näher an der echten Verteilung).")


if __name__ == "__main__":
    main()