from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)

SPLIT_DIRS = {
    "train": TRAIN_DIR,
    "validation": VALIDATION_DIR,
    "test": TEST_DIR,
}


def validate_split_structure() -> None:
    for split_name, split_dir in SPLIT_DIRS.items():
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing {split_name} directory: {split_dir}")
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")


def count_images_per_class(data_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        counts[class_name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if class_dir.exists() else 0
    return counts


def save_labels(save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, indent=4)


def get_split_dataset(
    split_name: str,
    *,
    shuffle: bool,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, list[str]]:
    validate_split_structure()
    if split_name not in SPLIT_DIRS:
        raise ValueError(f"Unknown split: {split_name}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIRS[split_name],
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=seed,
    )
    file_paths = list(getattr(dataset, "file_paths", []))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, file_paths


def get_image_paths_and_labels(data_dir: Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        class_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(str(path) for path in class_paths)
        labels.extend([label_index] * len(class_paths))
    if not paths:
        raise ValueError(f"No images found in {data_dir}")
    return paths, labels


def _load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, label


def create_dataset_from_paths(
    image_paths: list[str],
    labels: list[int],
    *,
    shuffle: bool,
    seed: int = SEED,
) -> tf.data.Dataset:
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length")
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def print_dataset_summary() -> None:
    validate_split_structure()
    total = 0
    for split_name, split_dir in SPLIT_DIRS.items():
        counts = count_images_per_class(split_dir)
        split_total = sum(counts.values())
        total += split_total
        print(f"\n{split_name.upper()} ({split_total})")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count}")
    print(f"\nTOTAL: {total}")


if __name__ == "__main__":
    print_dataset_summary()
