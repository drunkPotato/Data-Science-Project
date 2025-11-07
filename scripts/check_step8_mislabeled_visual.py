from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

LEAKS_CSV = Path("runs/qc/step7_split_leaks.csv")
OUT_DIR   = Path("runs/qc"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG   = OUT_DIR / "step8_mislabeled_sample.png"
OUT_CSV   = OUT_DIR / "step8_mislabeled_only.csv"
SAMPLE_N  = 24  # wie viele Paare wir visuell zeigen wollen (2 Bilder pro Paar)

def load_df():
    if not LEAKS_CSV.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {LEAKS_CSV}")
    df = pd.read_csv(LEAKS_CSV)
    # normalize boolean
    if df["same_label"].dtype != bool:
        df["same_label"] = df["same_label"].astype(str).str.lower().isin(["true","1","yes"])
    return df

def main():
    print("=== Schritt 8: Mislabeled-Check (visuelle Stichprobe) ===")
    df = load_df()
    # nur Paare mit unterschiedlichem Label
    mis = df[~df["same_label"]].copy()
    print(f"Gesamt potenziell mislabeled (train_label != test_label): {len(mis)}")

    # kleine Konfusions-Übersicht
    if len(mis):
        conf = Counter(zip(mis["train_label"], mis["test_label"]))
        print("Top-Konfusionen (train_label -> test_label, count):")
        for (tl, vl), c in conf.most_common(10):
            print(f"  {tl:>10} -> {vl:<10} : {c}")

        # exportiere komplette Liste
        mis.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"CSV mit allen mismatched-Fällen: {OUT_CSV.resolve()}")

        # Stichprobe für visuelle Kontrolle
        sample = mis.sample(min(SAMPLE_N, len(mis)), random_state=42)
        n = len(sample)
        cols = 2  # links train, rechts test
        rows = n  # jedes Paar ist eine Zeile

        # große, aber schmale Figure
        fig = plt.figure(figsize=(10, rows * 2.2), dpi=120)
        gs = fig.add_gridspec(rows, cols, wspace=0.02, hspace=0.2)

        for i, row in enumerate(sample.itertuples(index=False)):
            # train links
            try:
                img_t = Image.open(row.train_path).convert("RGB")
                ax = fig.add_subplot(gs[i, 0])
                ax.imshow(img_t)
                ax.set_title(f"TRAIN: {row.train_label}", fontsize=9)
                ax.set_xlabel(Path(row.train_path).name, fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax = fig.add_subplot(gs[i, 0]); ax.axis("off")
                ax.text(0.5, 0.5, f"Fehler:\n{e}", ha="center", va="center", fontsize=8)

            # test rechts
            try:
                img_v = Image.open(row.test_path).convert("RGB")
                ax = fig.add_subplot(gs[i, 1])
                ax.imshow(img_v)
                ax.set_title(f"TEST:  {row.test_label}", fontsize=9)
                ax.set_xlabel(Path(row.test_path).name, fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax = fig.add_subplot(gs[i, 1]); ax.axis("off")
                ax.text(0.5, 0.5, f"Fehler:\n{e}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        fig.savefig(OUT_IMG, bbox_inches="tight")
        plt.close(fig)
        print(f"Vorschau gespeichert: {OUT_IMG.resolve()}")
        print("👉 Öffne das PNG und schau, ob die Labels offensichtlich vertauscht wirken.")
    else:
        print("✅ Keine Paare mit unterschiedlichen Labels gefunden (nice!).")

if __name__ == "__main__":
    main()
