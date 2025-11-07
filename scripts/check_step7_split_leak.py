from pathlib import Path
import hashlib

print("✅ Script gestartet – working...")


DATASET_DIR = Path("data/raw/Faces_Dataset")
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def md5_of_file(p: Path, chunk_size=1 << 20) -> str:
    """Berechnet einen Hash, um identische Bilder zu erkennen."""
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_hashes(split: str) -> set:
    hashes = set()
    for label in LABELS:
        d = DATASET_DIR / split / label
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                try:
                    hashes.add(md5_of_file(p))
                except Exception as e:
                    print(f"Fehler beim Hashen: {p} ({e})")
    return hashes

if __name__ == "__main__":
    print("=== Schritt 7: Split-Leak Check (train vs test) ===")
    train_hashes = collect_hashes("train")
    test_hashes  = collect_hashes("test")

    print(f"Train Hashes: {len(train_hashes)}")
    print(f"Test  Hashes: {len(test_hashes)}")

    overlap = train_hashes & test_hashes
    print(f"Gemeinsame Hashes (Leak): {len(overlap)}")

    if overlap:
        print("⚠️  Achtung: identische Dateien in train und test gefunden!")
    else:
        print("✅ Kein Split-Leak mit exakten Duplikaten gefunden.")
