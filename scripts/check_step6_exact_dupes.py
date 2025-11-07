from pathlib import Path
import hashlib
from collections import defaultdict
import csv

DATASET_DIR = Path("data/raw/Faces_Dataset")
SPLITS = ["train", "test"]
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUT_DIR = Path("runs/qc"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "step6_exact_duplicates_report.csv"

def md5_of_file(p: Path, chunk_size=1<<20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_images():
    for split in SPLITS:
        for label in LABELS:
            d = DATASET_DIR / split / label
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    yield split, label, p

def main():
    print("=== Schritt 6 (exakt): Duplikate & Split-Leaks ===")
    print(f"Dataset root: {DATASET_DIR.resolve()}")

    records = []
    total = 0
    for split, label, path in iter_images():
        try:
            fhash = md5_of_file(path)
        except Exception as e:
            print(f"! MD5-Fehler bei {path}: {e}")
            continue
        records.append({"split": split, "label": label, "path": str(path), "md5": fhash})
        total += 1
        if total % 2000 == 0:
            print(f"  ... {total} Dateien gehasht")

    print(f"• Insgesamt gehasht: {total}")

    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)

    exact_pairs = []
    for group in by_md5.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    exact_pairs.append((group[i], group[j]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["split_a","label_a","path_a","split_b","label_b","path_b","same_split","same_label"])
        for a,b in exact_pairs:
            w.writerow([a["split"],a["label"],a["path"], b["split"],b["label"],b["path"],
                        a["split"]==b["split"], a["label"]==b["label"]])

    cross_split = sum(1 for a,b in exact_pairs if a["split"] != b["split"])

    print("\n=== Zusammenfassung Schritt 6 (exakt) ===")
    print(f"Exakte Duplikat-Paare: {len(exact_pairs)}")
    print(f"… davon split-übergreifend (train↔test): {cross_split}")
    print(f"CSV-Report: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
