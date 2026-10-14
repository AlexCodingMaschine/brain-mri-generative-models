"""
Misst die Bildcharakteristika des Brain Tumor MRI Datasets, damit der Bericht
keine Annahmen, sondern gemessene Werte enthält (Aufgabenpunkt: "Analyze the
characteristics of the selected medical images, including image dimensions,
intensity characteristics, anatomical content, and image-quality considerations").

Gemessen wird pro Split (Training/Testing) und Klasse:
  - Bildgrößen (Breite x Höhe): Häufigkeiten, Minimum, Maximum
  - Farbmodus (Graustufen "L" oder RGB) und ob RGB-Bilder tatsächlich nur Graustufen enthalten
  - Intensitäten (Graustufenwerte 0-255): Mittelwert, Standardabweichung, Anteil dunkler Pixel (< 10 = Hintergrund)
  - Beispielbilder pro Klasse (für die visuelle Beurteilung von Schnittebene und anatomischem Inhalt)

Ausführen im Projektordner (venv aktiviert):
    python analyze_dataset.py
oder mit anderem Datenpfad:
    python analyze_dataset.py --root data/brain_tumor_mri

Ergebnisse:
    outputs/dataset_statistics.csv       alle Messwerte pro Klasse
    outputs/dataset_samples.png          je 6 zufällige Bilder pro Klasse
    outputs/dataset_intensity_hist.png   Intensitätsverteilung pro Klasse
Die Zusammenfassung wird zusätzlich in der Konsole ausgegeben -- diese bitte kopieren und
im Chat zurückschicken, dann werden die echten Zahlen in den Bericht eingetragen.
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def analyze_class(paths):
    sizes, modes = Counter(), Counter()
    means, stds, dark_fracs = [], [], []
    hist = np.zeros(256, dtype=np.int64)
    rgb_total, rgb_gray = 0, 0
    for p in paths:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            if im.mode == "RGB":
                rgb_total += 1
                a = np.asarray(im)
                if np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]):
                    rgb_gray += 1
            g = np.asarray(im.convert("L"), dtype=np.uint8)
        means.append(float(g.mean()))
        stds.append(float(g.std()))
        dark_fracs.append(float((g < 10).mean()))
        hist += np.bincount(g.ravel(), minlength=256)
    widths = [s[0] for s in sizes.elements()]
    heights = [s[1] for s in sizes.elements()]
    return {
        "n_images": len(paths),
        "n_distinct_sizes": len(sizes),
        "most_common_sizes": "; ".join(f"{w}x{h} ({c})" for (w, h), c in sizes.most_common(5)),
        "min_size": f"{min(widths)}x{min(heights)}",
        "max_size": f"{max(widths)}x{max(heights)}",
        "modes": "; ".join(f"{m} ({c})" for m, c in modes.most_common()),
        "rgb_images_that_are_grayscale": f"{rgb_gray}/{rgb_total}",
        "mean_intensity": round(float(np.mean(means)), 1),
        "mean_std_intensity": round(float(np.mean(stds)), 1),
        "mean_dark_pixel_share": round(float(np.mean(dark_fracs)), 3),
    }, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/brain_tumor_mri", help="Ordner mit Training/ und Testing/")
    ap.add_argument("--out", default="outputs", help="Ausgabeordner")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    rows, hists, samples = [], {}, {}
    for split in ("Training", "Testing"):
        split_dir = root / split
        if not split_dir.exists():
            print(f"WARNUNG: {split_dir} nicht gefunden -- übersprungen")
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            paths = list_images(cls_dir)
            stats, hist = analyze_class(paths)
            rows.append({"split": split, "class": cls_dir.name, **stats})
            hists.setdefault(cls_dir.name, np.zeros(256, dtype=np.int64))
            hists[cls_dir.name] += hist
            if split == "Training":
                samples[cls_dir.name] = random.sample(paths, min(6, len(paths)))

    if not rows:
        raise SystemExit("Keine Bilder gefunden -- bitte --root prüfen.")

    # ---- CSV ----
    with open(out / "dataset_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Konsole ----
    print("\n=== BILDCHARAKTERISTIKA (bitte diesen Block kopieren) ===")
    for r in rows:
        print(f"\n[{r['split']}] {r['class']}: {r['n_images']} Bilder")
        print(f"  Größen: {r['n_distinct_sizes']} verschiedene | häufigste: {r['most_common_sizes']}")
        print(f"  Min/Max: {r['min_size']} / {r['max_size']}")
        print(f"  Farbmodus: {r['modes']} | RGB-Bilder mit identischen Kanälen (= Graustufen): {r['rgb_images_that_are_grayscale']}")
        print(f"  Intensität: Mittelwert {r['mean_intensity']}, Std {r['mean_std_intensity']}, dunkle Pixel (<10): {r['mean_dark_pixel_share']:.1%}")
    print("\nHinweis: Die Schnittebene (axial/koronal/sagittal) und der anatomische Inhalt lassen sich nicht automatisch messen --")
    print("bitte outputs/dataset_samples.png ansehen und hier kurz beschreiben, welche Ebenen du siehst.")

    # ---- Beispielbilder ----
    classes = list(samples.keys())
    fig, axes = plt.subplots(len(classes), 6, figsize=(12, 2.1 * len(classes)))
    axes = np.atleast_2d(axes)
    for i, cls in enumerate(classes):
        for j in range(6):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(samples[cls]):
                ax.imshow(Image.open(samples[cls][j]).convert("L"), cmap="gray")
            if j == 0:
                ax.set_title(cls, loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "dataset_samples.png", dpi=150)
    plt.close()

    # ---- Intensitätsverteilung ----
    plt.figure(figsize=(7, 3.5))
    for cls, h in hists.items():
        plt.plot(np.arange(256), h / h.sum(), label=cls, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Grauwert (0-255)")
    plt.ylabel("Anteil der Pixel (log)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "dataset_intensity_hist.png", dpi=150)
    plt.close()

    print(f"\nGespeichert: {out}/dataset_statistics.csv, dataset_samples.png, dataset_intensity_hist.png")


if __name__ == "__main__":
    main()
