from pathlib import Path
import shutil
import random

AI_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = AI_DIR / "dataset" / "resized"
OUTPUT_DIR = AI_DIR / "dataset" / "split"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

random.seed(RANDOM_SEED)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

train_dir = OUTPUT_DIR / "train"
test_dir = OUTPUT_DIR / "test"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

for class_folder in sorted(SOURCE_DIR.iterdir()):
    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    images = [
        img for img in class_folder.iterdir()
        if img.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    test_images = images[train_count:]

    train_class_dir = train_dir / class_name
    test_class_dir = test_dir / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy2(img, train_class_dir / img.name)

    for img in test_images:
        shutil.copy2(img, test_class_dir / img.name)

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("\nDataset splitting completed.")