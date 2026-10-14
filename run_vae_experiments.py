"""
Führt mehrere VAE-Experimente nacheinander aus und sammelt die Ergebnisse
in outputs/vae_experiments.csv.

Jedes Experiment ändert GENAU EINE Sache gegenüber "v1_baseline",
damit sich Effekte sauber zuordnen lassen (wichtig für den Report).

Beispiele:
    python run_vae_experiments.py --epochs 2          # schneller Testlauf (Minuten)
    python run_vae_experiments.py --epochs 30         # echter Lauf, alle Experimente
    python run_vae_experiments.py --only kl_0.1       # nur ein bestimmtes Experiment

Bereits fertige Experimente (Name steht schon in der CSV) werden übersprungen.
Du kannst den Lauf also jederzeit mit Strg+C abbrechen und später fortsetzen.
"""

import argparse
import csv
import os

from config import OUTPUT_DIR
from src.experiment import run_experiment

CSV_PATH = f"{OUTPUT_DIR}/vae_experiments.csv"

FIELDS = [
    "name", "latent_dim", "kl_weight", "sigma", "epochs",
    "psnr_noisy", "psnr_vae", "psnr_gauss", "psnr_median",
    "ssim_noisy", "ssim_vae", "ssim_gauss", "ssim_median", "minutes",
]

# Referenz: latent=128, kl=1.0, sigma=0.15 (entspricht deinem ersten Lauf)
EXPERIMENTS = [
    dict(name="v1_baseline", latent_dim=128, kl_weight=1.0,  sigma=0.15),
    # --- KL-Gewicht (Schärfe vs. Ordnung des latenten Raums) ---
    dict(name="kl_0.1",      latent_dim=128, kl_weight=0.1,  sigma=0.15),
    dict(name="kl_0.01",     latent_dim=128, kl_weight=0.01, sigma=0.15),
    # --- Latente Dimension (Kompression vs. Detail) ---
    dict(name="latent_32",   latent_dim=32,  kl_weight=1.0,  sigma=0.15),
    dict(name="latent_256",  latent_dim=256, kl_weight=1.0,  sigma=0.15),
    # --- Rauschstärke (Robustheit des Denoisings) ---
    dict(name="sigma_0.05",  latent_dim=128, kl_weight=1.0,  sigma=0.05),
    dict(name="sigma_0.30",  latent_dim=128, kl_weight=1.0,  sigma=0.30),
]


def already_done() -> set:
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def append_result(result: dict) -> None:
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(result)


def print_table() -> None:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("\n" + "=" * 100)
    print(f"{'Name':<13}{'lat':>5}{'kl':>7}{'sigma':>7} | "
          f"{'PSNR in':>8}{'VAE':>8}{'Gauss':>8}{'Median':>8} | "
          f"{'SSIM in':>8}{'VAE':>7}{'Gauss':>7}{'Median':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<13}{r['latent_dim']:>5}{r['kl_weight']:>7}{r['sigma']:>7} | "
              f"{float(r['psnr_noisy']):>8.2f}{float(r['psnr_vae']):>8.2f}"
              f"{float(r['psnr_gauss']):>8.2f}{float(r['psnr_median']):>8.2f} | "
              f"{float(r['ssim_noisy']):>8.3f}{float(r['ssim_vae']):>7.3f}"
              f"{float(r['ssim_gauss']):>7.3f}{float(r['ssim_median']):>7.3f}")
    print("=" * 100)
    print("PSNR in dB (höher = besser), SSIM 0..1 (höher = besser). 'in' = verrauschter Input ohne Entrauschen.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--only", nargs="*", default=None, help="nur diese Experiment-Namen")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done = already_done()

    for exp in EXPERIMENTS:
        if args.only and exp["name"] not in args.only:
            continue
        if exp["name"] in done:
            print(f"Überspringe {exp['name']} (schon in der CSV)")
            continue
        result = run_experiment(epochs=args.epochs, **exp)
        append_result(result)          # sofort speichern, falls du später abbrichst

    if os.path.exists(CSV_PATH):
        print_table()


if __name__ == "__main__":
    main()
