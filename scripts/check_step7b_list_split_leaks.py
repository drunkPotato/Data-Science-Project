from pathlib import Path
import hashlib
import csv

DATASET_DIR = Path("data/raw/Faces_Dataset")
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUT_DIR = Path("runs/qc"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "step7_split_leaks.csv"        # Liste der Leaks
OUT_TXT = OUT_DIR / "step7_split_leaks_summary.txt" # kurze Zusammenfassung

def md5_of_file(p: Path, chunk_size=1<<20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(split: str):
    """Map MD5 -> (label, path) nur für diesen Split."""
    m = {}
    for label in LABELS:
        d = DATASET_DIR / split / label
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                try:
                    m[md5_of_file(p)] = (label, str(p))
                except Exception:
                    pass
    return m

if __name__ == "__main__":
    print("=== Schritt 7b: Leaks auflisten (train vs test) ===")
    train_map = collect("train")
    test_map  = collect("test")

    inter = set(train_map.keys()) & set(test_map.keys())
    print(f"Train unique hashes: {len(train_map)}")
    print(f"Test  unique hashes: {len(test_map)}")
    print(f"Gemeinsame Hashes  : {len(inter)}")

    # CSV schreiben
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["md5","train_label","train_path","test_label","test_path","same_label"])
        for h in sorted(inter):
            tl, tp = train_map[h]
            vl, vp = test_map[h]
            w.writerow([h, tl, tp, vl, vp, tl == vl])

    # kurze Text-Zusammenfassung
    same_label = 0
    diff_label = 0
    for h in inter:
        tl, _ = train_map[h]
        vl, _ = test_map[h]
        if tl == vl:
            same_label += 1
        else:
            diff_label += 1

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("Split-Leak Summary\n")
        f.write(f"train hashes    : {len(train_map)}\n")
        f.write(f"test hashes     : {len(test_map)}\n")
        f.write(f"overlap (leaks) : {len(inter)}\n")
        f.write(f"same label      : {same_label}\n")
        f.write(f"diff label      : {diff_label}\n")

    print(f"CSV-Report   -> {OUT_CSV.resolve()}")
    print(f"Summary note -> {OUT_TXT.resolve()}")
    if diff_label:
        print(f"⚠️  {diff_label} Leaks haben unterschiedliche Labels (mögliche Mislables).")
    else:
        print("✅ Alle Leaks haben dasselbe Label (nur Split-Leak).")
