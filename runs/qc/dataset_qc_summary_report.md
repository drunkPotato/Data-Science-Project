# Dataset Quality Control Summary

_Automatisch erstellt am 2025-11-07 15:18_


## Struktur, Formate und Lesbarkeit

- ✅ Alle 7 Emotion-Ordner vorhanden

- ✅ Bilder lesbar und nicht korrupt

- ✅ Konsistentes Format (.png)


## Klassenverteilung
⚠️ Keine Statistik-Datei gefunden


## Doppelte Bilder
- Gefundene exakte Duplikat-Paare: **2418**


## Split-Leaks

- Gefundene train↔test-Leaks: **531**

- Davon mit unterschiedlichen Labels: **20**

⚠️ Einige Bilder haben widersprüchliche Emotion-Labels.


## Potenziell mislabeled Images
- Anzahl: **20**

- Beispiel-Vorschau: `runs/qc/step8_mislabeled_sample.png`

👉 Manuelle Überprüfung empfohlen.


## Gesamtbewertung

- 🧩 Datensatz ist größtenteils konsistent und vollständig.

- 🚨 Es gibt einige doppelte und falsch gelabelte Bilder, die vor Training entfernt/angepasst werden sollten.

- 📊 Alle Ergebnisse liegen im Ordner `runs/qc` vor.
