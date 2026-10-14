"""
Zentrale Konfiguration für das Brain-MRI Generative-AI-Projekt.
Alle Hyperparameter und Pfade werden hier gesammelt, damit sie
nicht über mehrere Skripte verstreut sind.
"""

import torch

# ---------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------
DATA_ROOT = "data/brain_tumor_mri"       # nach kagglehub-Download hier verlinken/kopieren
TRAIN_DIR = f"{DATA_ROOT}/Training"
TEST_DIR = f"{DATA_ROOT}/Testing"
OUTPUT_DIR = "outputs"

# Für Phase 1 (GAN): nur eine Klasse verwenden, um Mode Collapse zu vermeiden.
# Optionen: "glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"
GAN_TARGET_CLASS = "no_tumor"

# ---------------------------------------------------------------------
# Bildverarbeitung
# ---------------------------------------------------------------------
IMAGE_SIZE = 64          # Start klein halten, später z.B. auf 128 erhöhen
CHANNELS = 1             # MRT ist eigentlich einkanalig (Graustufen)

# ---------------------------------------------------------------------
# Rauschen für den VAE (Denoising-Aufgabe)
# ---------------------------------------------------------------------
NOISE_TYPE = "gaussian"  # Optionen: "gaussian", "salt_pepper", "blur"
GAUSSIAN_NOISE_SIGMA = 0.15   # Stärke des Rauschens (0.0 - 1.0 Skala)

# ---------------------------------------------------------------------
# Training – allgemein
# ---------------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# GAN Hyperparameter
# ---------------------------------------------------------------------
GAN_LATENT_DIM = 100
GAN_LR = 2e-4
GAN_BETA1 = 0.5           # Adam-Beta1, Standardwert für DCGAN-Training
GAN_EPOCHS = 100

# ---------------------------------------------------------------------
# VAE Hyperparameter
# ---------------------------------------------------------------------
VAE_LATENT_DIM = 128
VAE_LR = 1e-3
VAE_EPOCHS = 50
VAE_KL_WEIGHT = 1.0        # Gewichtung der KL-Divergenz im Loss (Beta-VAE-Style)

# ---------------------------------------------------------------------
# Diffusion Model Hyperparameter
# ---------------------------------------------------------------------
DIFFUSION_TIMESTEPS = 200   # Anzahl Rausch-Schritte T (Original-Paper nutzt 1000, wir halten es klein für CPU)
DIFFUSION_BASE_CHANNELS = 32
DIFFUSION_LR = 2e-4
DIFFUSION_EPOCHS = 40
# None = alle Klassen nutzen (mehr Daten). Ein Klassenname wie beim GAN würde nur
# diese eine Klasse nutzen -- für den Datenmengen-Test lassen wir es bei None.
DIFFUSION_TARGET_CLASS = None