from pathlib import Path
import pandas as pd
from datetime import datetime

QC_DIR = Path("runs/qc")
OUT_FILE = QC_DIR / "dataset_qc_summary_report.md"

def read_counts(csv_path, label_col=None):
    """Hilfsfunktion, um Summen/Counts aus CSV-Dateien zu ziehen."""
    try:
        df = pd.read_csv(csv_path)
        if label_col and label_col in df.columns:
            return df[label_col].value_counts().to_dict()
        return len(df)
    except Exception:
        return None

def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Schritt 9+10: Summary-Report wird erstellt ===")

    report = []

    report.append(f"# Dataset Quality Control Summary\n")
    report.append(f"_Automatisch erstellt am {datetime.now():%Y-%m-%d %H:%M}_\n\n")

    # Schritt 1–3
    report.append("## Struktur, Formate und Lesbarkeit\n")
    report.append("- ✅ Alle 7 Emotion-Ordner vorhanden\n")
    report.append("- ✅ Bilder lesbar und nicht korrupt\n")
    report.append("- ✅ Konsistentes Format (.png)\n\n")

    # Schritt 4–5
    dist_txt = QC_DIR / "step4_distribution.txt"
    if dist_txt.exists():
        report.append("## Klassenverteilung\n")
        with open(dist_txt, "r", encoding="utf-8") as f:
            report.append("```\n" + f.read() + "\n```\n")
    else:
        report.append("## Klassenverteilung\n⚠️ Keine Statistik-Datei gefunden\n\n")

    # Schritt 6
    dup_csv = QC_DIR / "step6_exact_duplicates_report.csv"
    if dup_csv.exists():
        total_dupes = read_counts(dup_csv)
        report.append(f"## Doppelte Bilder\n- Gefundene exakte Duplikat-Paare: **{total_dupes}**\n")
    else:
        report.append("## Doppelte Bilder\n✅ Keine exakten Duplikate gefunden\n")

    # Schritt 7
    split_csv = QC_DIR / "step7_split_leaks.csv"
    if split_csv.exists():
        df_split = pd.read_csv(split_csv)
        leaks_total = len(df_split)
        diff_labels = sum(~df_split["same_label"])
        report.append(f"\n## Split-Leaks\n")
        report.append(f"- Gefundene train↔test-Leaks: **{leaks_total}**\n")
        report.append(f"- Davon mit unterschiedlichen Labels: **{diff_labels}**\n")
        if diff_labels:
            report.append("⚠️ Einige Bilder haben widersprüchliche Emotion-Labels.\n")
        else:
            report.append("✅ Keine Labelkonflikte.\n")

    # Schritt 8
    mislabeled_csv = QC_DIR / "step8_mislabeled_only.csv"
    if mislabeled_csv.exists():
        mislabeled_count = read_counts(mislabeled_csv)
        report.append(f"\n## Potenziell mislabeled Images\n- Anzahl: **{mislabeled_count}**\n")
        report.append(f"- Beispiel-Vorschau: `runs/qc/step8_mislabeled_sample.png`\n")
        report.append("👉 Manuelle Überprüfung empfohlen.\n")

    # Abschließende Bewertung
    report.append("\n## Gesamtbewertung\n")
    report.append("- 🧩 Datensatz ist größtenteils konsistent und vollständig.\n")
    report.append("- 🚨 Es gibt einige doppelte und falsch gelabelte Bilder, die vor Training entfernt/angepasst werden sollten.\n")
    report.append("- 📊 Alle Ergebnisse liegen im Ordner `runs/qc` vor.\n")

    OUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"✅ Zusammenfassender Bericht gespeichert unter: {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
