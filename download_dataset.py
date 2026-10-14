"""
Lädt das Brain Tumor MRI Dataset von Kaggle herunter und kopiert es
in die vom Projekt erwartete Ordnerstruktur (data/brain_tumor_mri/...).

Voraussetzung: kaggle.json liegt unter ~/.kaggle/kaggle.json
(siehe Kaggle -> Account Settings -> API -> Create New Token)

Ausführen mit:
    python download_dataset.py
"""

import shutil
from pathlib import Path

import kagglehub

TARGET_DIR = Path("data/brain_tumor_mri")


def main() -> None:
    print("Lade Dataset von Kaggle herunter (nutzt lokalen Cache, falls vorhanden)...")
    cache_path = Path(kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri"))
    print(f"Dataset liegt im Cache unter: {cache_path}")

    if TARGET_DIR.exists():
        print(f"Zielordner {TARGET_DIR} existiert bereits – überspringe Kopiervorgang.")
    else:
        TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Kopiere nach {TARGET_DIR} ...")
        shutil.copytree(cache_path, TARGET_DIR)

    # Kurze Übersicht, was tatsächlich angekommen ist
    print("\nGefundene Ordnerstruktur:")
    for item in sorted(TARGET_DIR.rglob("*")):
        if item.is_dir():
            n_images = len(list(item.glob("*.jpg"))) + len(list(item.glob("*.png")))
            if n_images > 0:
                print(f"  {item.relative_to(TARGET_DIR)}: {n_images} Bilder")


if __name__ == "__main__":
    main()