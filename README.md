# Synthetic Medical Image Generation: GAN, VAE and Diffusion Model on Brain MRI

Implementation for the project report of the IU course **DLMAIPGDL01 – Project: Generative Deep Learning** (Task 2).
Three generative models were implemented, trained and evaluated on 64×64 grayscale brain MRI slices:

- **GAN (DCGAN)** – generation of synthetic `no_tumor` images (Version 1 and Version 2 with label smoothing)
- **VAE** – denoising of artificially noised images, systematic hyperparameter experiments
- **Diffusion model (simplified DDPM)** – generation of synthetic images, two training runs

All results, figures and discussions are documented in the project report.

## Dataset

Brain Tumor Classification (MRI), Kaggle: `sartajbhuvaji/brain-tumor-classification-mri`
(Bhuvaji et al., 2020, DOI: 10.34740/KAGGLE/DSV/1183165). The dataset is **not** included in this repository.

After downloading, place it so that the following structure exists:

```
data/brain_tumor_mri/
├── Training/   (glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor)
└── Testing/    (same four classes)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Training was carried out on CPU. All hyperparameters are collected in `config.py` (seed 42).

## Repository structure

```
config.py                 central configuration (paths, hyperparameters)
src/dataset.py            dataset class (grayscale, 64×64, [0, 1])
src/models_gan.py         DCGAN generator and discriminator
src/models_vae.py         VAE and loss function
src/models_diffusion.py   simplified DDPM (noise schedule, U-Net)
analyze_dataset.py        image statistics (Appendix B of the report)
train_gan.py              GAN Version 1
train_gan_v2.py           GAN Version 2 (label smoothing 0.9)
check_gan_losses.py       loss measurement on the saved final GAN models
compute_fid.py            FID evaluation of both GAN versions
train_diffusion.py        diffusion model training
<VAE-Skript>.py           VAE experiment series        (Dateiname anpassen)
<Klassifikator-Skript>.py data augmentation experiment (Dateiname anpassen)
```

## Usage (examples)

```bash
python analyze_dataset.py
python train_gan.py
python train_gan_v2.py
python check_gan_losses.py          # Version 1
python check_gan_losses.py --v2     # Version 2
python compute_fid.py
python train_diffusion.py
```

Outputs (sample images, checkpoints, CSV results) are written to `outputs/`.

## Note on AI assistance

The code was developed with the support of Claude (Anthropic) and executed and verified locally by the author.
Details are given in the AI Transparency Statement of the project report.
