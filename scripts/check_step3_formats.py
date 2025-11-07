from pathlib import Path

DATASET_DIR = Path("data/raw/Faces_Dataset")
SPLITS = ["train", "test"]

def check_image_formats(root: Path):
    formats = {}

    for split in SPLITS:
        split_path = root / split
        print(f"\nSplit: {split}")
        for emotion_dir in split_path.rglob("*"):  # durchsucht ALLE Unterordner
            if emotion_dir.is_file():
                ext = emotion_dir.suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1

    print("\n=== Schritt 3: Bildformate ===")
    if not formats:
        print("⚠️ Keine Bilddateien gefunden – prüfe den Pfad oder Dateitypen.")
        return

    total = sum(formats.values())
    for ext, count in sorted(formats.items(), key=lambda x: -x[1]):
        print(f"{ext or '(ohne Endung)'}: {count} Dateien ({count/total*100:.2f}%)")

    if len(formats) == 1:
        print("✅ Alle Bilder haben dasselbe Format.")
    else:
        print("⚠️ Mehrere Formate gefunden – prüfe Konsistenz und Konvertierung.")

if __name__ == "__main__":
    check_image_formats(DATASET_DIR)
