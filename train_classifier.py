"""
Beantwortet die Kernfrage der Aufgabenstellung: Helfen die GAN-generierten
Bilder als Data Augmentation für eine nachgelagerte Klassifikationsaufgabe?

Zwei Durchläufe, sonst IDENTISCH:
  "baseline" : nur echte Trainingsbilder (no_tumor bleibt unterrepräsentiert: 395 vs. ~825)
  "augmented": echte + synthetische no_tumor-Bilder (Klassen ausgeglichen)

Ausgewertet wird NUR auf dem echten, unberührten Testing/-Ordner -- die
synthetischen Bilder fließen dort niemals ein.

Voraussetzung: generate_synthetic.py wurde bereits ausgeführt.

Ausführen mit:
    python train_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import classification_report

from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, CHANNELS, BATCH_SIZE, SEED, DEVICE, OUTPUT_DIR
from src.dataset import BrainMRIDataset
from src.classifier import SimpleClassifier

CLASSES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]
EPOCHS = 15


class LabeledDataset(Dataset):
    """Hängt an jedes Bild eines BrainMRIDataset ein festes Klassen-Label an."""
    def __init__(self, base_dataset, label: int):
        self.base = base_dataset
        self.label = label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx], self.label


class SyntheticDataset(Dataset):
    """Verpackt die gespeicherten synthetischen Tensoren als Dataset mit Label."""
    def __init__(self, tensor_path: str, label: int):
        self.images = torch.load(tensor_path)
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.label


def build_train_set(use_synthetic: bool) -> ConcatDataset:
    parts = []
    for i, cls in enumerate(CLASSES):
        parts.append(LabeledDataset(BrainMRIDataset(TRAIN_DIR, IMAGE_SIZE, CHANNELS, target_class=cls), i))
    if use_synthetic:
        no_tumor_idx = CLASSES.index("no_tumor")
        parts.append(SyntheticDataset(f"{OUTPUT_DIR}/synthetic_no_tumor.pt", no_tumor_idx))
    return ConcatDataset(parts)


def build_test_set() -> ConcatDataset:
    parts = []
    for i, cls in enumerate(CLASSES):
        parts.append(LabeledDataset(BrainMRIDataset(TEST_DIR, IMAGE_SIZE, CHANNELS, target_class=cls), i))
    return ConcatDataset(parts)


def train_one_run(name: str, use_synthetic: bool, test_loader: DataLoader) -> dict:
    torch.manual_seed(SEED)
    train_ds = build_train_set(use_synthetic)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"\n=== {name}: {len(train_ds)} Trainingsbilder "
         f"({'inkl.' if use_synthetic else 'OHNE'} synthetische no_tumor-Bilder) ===")

    model = SimpleClassifier(len(CLASSES), CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoche {epoch+1}/{EPOCHS} | Train-Loss {total_loss/len(train_loader):.3f}")

    # ---- Auswertung auf dem ECHTEN Testset ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    report = classification_report(all_labels, all_preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(f"\n  Ergebnis auf dem echten Testset ({name}):")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))
    return report


def main() -> None:
    test_ds = build_test_set()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test-Set (unberührt, nur echte Bilder): {len(test_ds)} Bilder")

    baseline = train_one_run("Baseline (nur echte Daten)", use_synthetic=False, test_loader=test_loader)
    augmented = train_one_run("Augmentiert (+ synthetische no_tumor-Bilder)", use_synthetic=True, test_loader=test_loader)

    print("\n" + "=" * 70)
    print(f"{'Metrik':<28}{'Baseline':>18}{'Augmentiert':>18}")
    print("-" * 70)
    print(f"{'Gesamt-Accuracy':<28}{baseline['accuracy']:>18.3f}{augmented['accuracy']:>18.3f}")
    for cls in CLASSES:
        b, a = baseline[cls], augmented[cls]
        print(f"{'  ' + cls + ' Recall':<28}{b['recall']:>18.3f}{a['recall']:>18.3f}")
        print(f"{'  ' + cls + ' Precision':<28}{b['precision']:>18.3f}{a['precision']:>18.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
