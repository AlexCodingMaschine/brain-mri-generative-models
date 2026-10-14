"""
PyTorch Dataset-Klassen für das Brain-MRI-Projekt.

Zwei Varianten:
- BrainMRIDataset:        liefert nur Original-Bilder (für GAN-Training)
- BrainMRIDenoisingDataset: liefert (verrauscht, original)-Paare (für VAE-Training)
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.noise import apply_degradation


def get_base_transform(image_size: int, channels: int) -> transforms.Compose:
    """Standard-Preprocessing: Resize, Graustufen, Tensor, Normalisierung auf [0,1]."""
    t = [transforms.Resize((image_size, image_size))]
    if channels == 1:
        t.append(transforms.Grayscale(num_output_channels=1))
    t.append(transforms.ToTensor())  # skaliert automatisch auf [0, 1]
    return transforms.Compose(t)


def _collect_image_paths(root_dir: str, target_class: str | None = None) -> list[str]:
    """Sammelt alle Bildpfade, optional gefiltert auf eine einzelne Klasse (Unterordner)."""
    root = Path(root_dir)
    paths = []
    classes = [target_class] if target_class else [d.name for d in root.iterdir() if d.is_dir()]
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Klassenordner nicht gefunden: {cls_dir}")
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(str(cls_dir / fname))
    return paths


class BrainMRIDataset(Dataset):
    """Für das GAN: liefert nur Bilder einer (typischerweise) einzelnen Klasse."""

    def __init__(self, root_dir: str, image_size: int, channels: int,
                 target_class: str | None = None):
        self.paths = _collect_image_paths(root_dir, target_class)
        self.transform = get_base_transform(image_size, channels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class BrainMRIDenoisingDataset(Dataset):
    """Für den VAE: liefert (verrauschtes Bild, Original-Bild)-Paare.

    Alle Klassen können hier genutzt werden, da es beim Denoising nicht um
    eine enge Datenverteilung geht wie beim GAN.
    """

    def __init__(self, root_dir: str, image_size: int, channels: int,
                 noise_type: str = "gaussian", noise_kwargs: dict | None = None,
                 target_class: str | None = None):
        self.paths = _collect_image_paths(root_dir, target_class)
        self.transform = get_base_transform(image_size, channels)
        self.noise_type = noise_type
        self.noise_kwargs = noise_kwargs or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.transform(img)
        noisy = apply_degradation(clean, self.noise_type, **self.noise_kwargs)
        return noisy, clean
