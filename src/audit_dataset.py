from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASS_NAMES, IMAGE_EXTENSIONS, OUTPUTS_DIR, RAW_DATASET_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size + 1, size))
        pixels = np.asarray(gray)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def main() -> None:
    audit_dir = OUTPUTS_DIR / "dataset_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATASET_DIR / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "class_name": class_name,
                        "path": str(path),
                        "sha256": sha256(path),
                        "dhash": difference_hash(path),
                    }
                )

    exact_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        exact_groups[record["sha256"]].append(record)

    exact_path = audit_dir / "exact_duplicates.csv"
    with open(exact_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["group_id", "class_name", "path", "cross_class_duplicate"])
        group_id = 0
        for group in exact_groups.values():
            if len(group) < 2:
                continue
            group_id += 1
            cross_class = len({item["class_name"] for item in group}) > 1
            for item in group:
                writer.writerow([group_id, item["class_name"], item["path"], cross_class])

    near_path = audit_dir / "potential_near_duplicates.csv"
    with open(near_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path_1", "class_1", "path_2", "class_2", "dhash_distance"])
        for index, first in enumerate(records):
            for second in records[index + 1 :]:
                if first["sha256"] == second["sha256"]:
                    continue
                distance = hamming_distance(first["dhash"], second["dhash"])
                if distance <= 4:
                    writer.writerow(
                        [
                            first["path"],
                            first["class_name"],
                            second["path"],
                            second["class_name"],
                            distance,
                        ]
                    )

    print(f"Exact duplicate report: {exact_path}")
    print(f"Potential near-duplicate report: {near_path}")
    print("Review potential near-duplicates manually before splitting.")


if __name__ == "__main__":
    main()
