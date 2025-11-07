from pathlib import Path
from collections import defaultdict, Counter
import sys

# === Einstellungen ===
DATASET_DIR = Path("data/raw/Faces_Dataset")
SPLITS = ["train", "test"]
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

IMBALANCE_RATIO_THRESHOLD = 3.0   # max/min > 3  => starke Imbalance
IMBALANCE_SHARE_THRESHOLD = 0.45  # eine Klasse > 45% => starke Imbalance

def count_images(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return sum(1 for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

def main():
    print("=== Schritt 4: Verteilung je Emotion & Split ===", flush=True)
    print(f"Dataset root: {DATASET_DIR.resolve()}", flush=True)

    if not DATASET_DIR.exists():
        print("❌ Dataset-Pfad existiert nicht. Bitte DATASET_DIR prüfen.", flush=True)
        sys.exit(1)

    split_counts = {}
    overall = Counter()

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        if not split_dir.is_dir():
            print(f"⚠️ Split-Ordner fehlt: {split_dir}", flush=True)
            split_counts[split] = {label: 0 for label in LABELS}
            continue

        counts = {}
        for label in LABELS:
            n = count_images(split_dir / label)
            counts[label] = n
            overall[label] += n
        split_counts[split] = counts

    # Ausgabe je Split + Imbalance-Check
    for split, counts in split_counts.items():
        total = sum(counts.values())
        print(f"\nSplit: {split} (Total: {total})", flush=True)
        if total == 0:
            print("  (Keine Bilder gefunden.)", flush=True)
            continue

        print(f"{'Label':<12}{'Count':>8}{'Share':>10}", flush=True)
        for label in LABELS:
            n = counts[label]
            share = n / total
            print(f"{label:<12}{n:>8}{share:>9.2%}", flush=True)
        print("-" * 30, flush=True)

        # Schritt 5: starke Imbalance flaggen
        nonzero = [c for c in counts.values() if c > 0]
        ratio_flag = False
        share_flag = False
        if nonzero:
            ratio = (max(nonzero) / min(nonzero)) if len(nonzero) > 1 else 1.0
            if ratio > IMBALANCE_RATIO_THRESHOLD:
                ratio_flag = True
        dominant_label = max(counts, key=counts.get) if counts else None
        dominant_share = counts[dominant_label] / total if dominant_label else 0.0
        if dominant_share > IMBALANCE_SHARE_THRESHOLD:
            share_flag = True

        if ratio_flag or share_flag:
            msg = ["⚠️  Imbalance erkannt:"]
            if ratio_flag:
                msg.append(f"max/min-Ratio > {IMBALANCE_RATIO_THRESHOLD}")
            if share_flag:
                msg.append(f"dominante Klasse '{dominant_label}' Anteil {dominant_share:.2%} > {IMBALANCE_SHARE_THRESHOLD:.0%}")
            print(" | ".join(msg), flush=True)
        else:
            print("✅ Keine starke Imbalance nach den definierten Schwellwerten.", flush=True)

    # Gesamtübersicht
    grand_total = sum(overall.values())
    print("\n=== Gesamt (train+test) ===", flush=True)
    print(f"{'Label':<12}{'Count':>8}{'Share':>10}", flush=True)
    for label in LABELS:
        n = overall[label]
        share = (n / grand_total) if grand_total else 0.0
        print(f"{label:<12}{n:>8}{share:>9.2%}", flush=True)
    print("-" * 30, flush=True)
    print(f"{'TOTAL':<12}{grand_total:>8}{'100.00%':>10}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}", flush=True)
        raise
