from pathlib import Path
from PIL import Image

AI_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = AI_DIR / "dataset" / "original"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

total_images = 0

print("Checking dataset...\n")

for class_folder in sorted(DATASET_DIR.iterdir()):
    if not class_folder.is_dir():
        continue

    images = [
        img for img in class_folder.iterdir()
        if img.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{class_folder.name}: {len(images)} images")
    total_images += len(images)

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"  Unreadable image: {img_path.name} | Error: {e}")

print(f"\nTotal images: {total_images}")
print("Dataset checking completed.")