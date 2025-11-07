from pathlib import Path
from PIL import Image

# Hauptordner deines Datensatzes
DATASET_DIR = Path("data/raw/Faces_Dataset")
SPLITS = ["train", "test"]

def check_images_readable(root: Path):
    total_images = 0
    corrupted = []

    for split in SPLITS:
        split_path = root / split
        for emotion_dir in split_path.iterdir():
            if emotion_dir.is_dir():
                for img_path in emotion_dir.glob("*"):
                    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                        continue
                    total_images += 1
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # prüft nur, ob Datei technisch ok ist
                    except Exception as e:
                        corrupted.append((img_path, str(e)))

    print("\n=== Schritt 2: Bildprüfung ===")
    print(f"Gesamtzahl Bilder: {total_images}")
    if corrupted:
        print(f"❌ {len(corrupted)} beschädigte oder unlesbare Bilder gefunden:")
        for path, err in corrupted[:10]:  # nur erste 10 anzeigen
            print(f" - {path} → {err}")
    else:
        print("✅ Alle überprüften Bilder sind lesbar und intakt.")

if __name__ == "__main__":
    check_images_readable(DATASET_DIR)
