from __future__ import annotations

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    RAW_DATASET_DIR,
    SEED,
    SPLIT_DATASET_DIR,
    TEST_DIR,
    TEST_RATIO,
    TRAIN_DIR,
    TRAIN_RATIO,
    VALIDATION_DIR,
    VALIDATION_RATIO,
)

CLEAR_EXISTING_SPLIT = True


def _safe_destination(source: Path, class_root: Path, destination_dir: Path) -> Path:
    relative = source.relative_to(class_root)
    flattened = "__".join(relative.parts)
    destination = destination_dir / flattened
    counter = 1
    while destination.exists():
        destination = destination_dir / f"{source.stem}_{counter}{source.suffix.lower()}"
        counter += 1
    return destination


def split_dataset() -> None:
    if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    if CLEAR_EXISTING_SPLIT and SPLIT_DATASET_DIR.exists():
        shutil.rmtree(SPLIT_DATASET_DIR)

    for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "validation": 0, "test": 0}
    print("Creating fixed 70/10/20 split per class...\n")

    for class_name in CLASS_NAMES:
        class_root = RAW_DATASET_DIR / class_name
        if not class_root.exists():
            raise FileNotFoundError(f"Missing raw class folder: {class_root}")

        images = sorted(
            path
            for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in {class_root}")

        test_count = round(len(images) * TEST_RATIO)
        validation_count = round(len(images) * VALIDATION_RATIO)

        development_images, test_images = train_test_split(
            images,
            test_size=test_count,
            random_state=SEED,
            shuffle=True,
        )
        train_images, validation_images = train_test_split(
            development_images,
            test_size=validation_count,
            random_state=SEED,
            shuffle=True,
        )

        expected_train = len(images) - validation_count - test_count
        if len(train_images) != expected_train:
            raise RuntimeError("Unexpected split count")

        split_lists = {
            "train": (train_images, TRAIN_DIR / class_name),
            "validation": (validation_images, VALIDATION_DIR / class_name),
            "test": (test_images, TEST_DIR / class_name),
        }

        for split_name, (split_images, destination_dir) in split_lists.items():
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in split_images:
                destination = _safe_destination(source, class_root, destination_dir)
                shutil.copy2(source, destination)
            totals[split_name] += len(split_images)

        print(
            f"{class_name}: train={len(train_images)}, "
            f"validation={len(validation_images)}, test={len(test_images)}"
        )

    print("\nSplit complete.")
    print(f"Training: {totals['train']}")
    print(f"Validation: {totals['validation']}")
    print(f"Testing: {totals['test']}")
    print(f"Total: {sum(totals.values())}")


if __name__ == "__main__":
    split_dataset()
