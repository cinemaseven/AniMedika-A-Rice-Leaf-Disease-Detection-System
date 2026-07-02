from pathlib import Path
from PIL import Image
import shutil

AI_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = AI_DIR / "dataset" / "original"
OUTPUT_DIR = AI_DIR / "dataset" / "resized"

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_folder in sorted(SOURCE_DIR.iterdir()):
    if not class_folder.is_dir():
        continue

    output_class_folder = OUTPUT_DIR / class_folder.name
    output_class_folder.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in class_folder.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            save_path = output_class_folder / f"{image_path.stem}.jpg"
            img.save(save_path, "JPEG", quality=95)

            count += 1

        except Exception as e:
            print(f"Skipped {image_path}: {e}")

    print(f"{class_folder.name}: {count} resized images")

print("\nImage resizing completed.")