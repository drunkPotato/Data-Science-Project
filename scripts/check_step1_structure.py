from pathlib import Path

# Pfad zum Hauptdatensatz (laut Datasets.md)
DATASET_DIR = Path("data/raw/Faces_Dataset")

# Die vorhandenen Splits (laut Dokumentation: train und test)
EXPECTED_SPLITS = ["train", "test"]

# 7 Emotion-Kategorien (wie in der Beschreibung)
EXPECTED_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"
]

def check_structure(root: Path):
    problems = []
    print(f"Dataset root: {root.resolve()}\n")

    # Prüfe alle Splits
    for split in EXPECTED_SPLITS:
        split_path = root / split
        if not split_path.is_dir():
            problems.append(f"FEHLT: Split-Ordner '{split}' existiert nicht.")
            continue

        # Finde vorhandene Emotion-Unterordner
        found_labels = sorted([p.name for p in split_path.iterdir() if p.is_dir()])
        missing = sorted(set(EXPECTED_LABELS) - set(found_labels))
        extra = sorted(set(found_labels) - set(EXPECTED_LABELS))

        print(f"Split: {split}")
        print(f"  Gefundene Labels ({len(found_labels)}): {found_labels}")
        if missing:
            problems.append(f"  FEHLENDE Labels in '{split}': {missing}")
        if extra:
            problems.append(f"  UNERWARTETE Labels in '{split}': {extra}")
        print()

    # Zusammenfassung
    print("=== Zusammenfassung Schritt 1 ===")
    if problems:
        print("PROBLEME GEFUNDEN:")
        for p in problems:
            print(" -", p)
    else:
        print("Alles gut ✅ Alle Splits und Emotion-Ordner sind vorhanden.")

if __name__ == "__main__":
    check_structure(DATASET_DIR)
