from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import (
    AUDIT_DIR,
    AUDIT_SUMMARY_PATH,
    CLASS_NAMES,
    DATASET_GROUPS_PATH,
    EXACT_DUPLICATES_PATH,
    IMAGE_EXTENSIONS,
    IMAGE_INVENTORY_PATH,
    NEAR_DUPLICATE_REVIEW_PATH,
    NEAR_DUPLICATE_THRESHOLD,
    RAW_DATASET_DIR,
    create_required_folders,
)


VALID_DECISIONS = {"yes", "no", ""}


@dataclass(frozen=True)
class ImageRecord:
    class_name: str
    path: Path
    relative_path: str
    sha256: str
    dhash: int
    width: int
    height: int
    mode: str


class UnionFind:
    """Small deterministic disjoint-set structure for duplicate groups."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        rank_first = self.rank[root_first]
        rank_second = self.rank[root_second]

        if rank_first < rank_second:
            root_first, root_second = root_second, root_first

        self.parent[root_second] = root_first

        if rank_first == rank_second:
            self.rank[root_first] += 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
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


def _normalise_decision(value: object) -> str:
    decision = str(value or "").strip().lower()

    aliases = {
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "group": "yes",
        "same": "yes",
        "n": "no",
        "false": "no",
        "0": "no",
        "different": "no",
        "review": "",
        "pending": "",
    }

    decision = aliases.get(decision, decision)

    if decision not in VALID_DECISIONS:
        raise ValueError(
            "Invalid near-duplicate decision "
            f"'{value}'. Use yes, no, or leave it blank."
        )

    return decision


def _pair_key(first_relative: str, second_relative: str) -> tuple[str, str]:
    return tuple(sorted((first_relative, second_relative)))


def _load_existing_near_review() -> dict[tuple[str, str], dict[str, str]]:
    if not NEAR_DUPLICATE_REVIEW_PATH.exists():
        return {}

    decisions: dict[tuple[str, str], dict[str, str]] = {}

    with NEAR_DUPLICATE_REVIEW_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            first = str(row.get("relative_path_1", "")).strip()
            second = str(row.get("relative_path_2", "")).strip()

            if not first or not second:
                continue

            decisions[_pair_key(first, second)] = {
                "decision": _normalise_decision(row.get("decision", "")),
                "notes": str(row.get("notes", "")).strip(),
            }

    return decisions


def _iter_raw_images() -> list[tuple[str, Path]]:
    images: list[tuple[str, Path]] = []

    for class_name in CLASS_NAMES:
        class_dir = RAW_DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing raw class folder: {class_dir}")

        class_images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not class_images:
            raise ValueError(f"No supported image files found in {class_dir}")

        images.extend((class_name, path) for path in class_images)

    return images


def _scan_images() -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    unreadable_rows: list[dict[str, str]] = []

    for class_name, path in _iter_raw_images():
        relative_path = path.relative_to(RAW_DATASET_DIR).as_posix()

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode

            record = ImageRecord(
                class_name=class_name,
                path=path.resolve(),
                relative_path=relative_path,
                sha256=sha256(path),
                dhash=difference_hash(path),
                width=int(width),
                height=int(height),
                mode=str(mode),
            )
            records.append(record)

        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
        ) as error:
            unreadable_rows.append(
                {
                    "class_name": class_name,
                    "relative_path": relative_path,
                    "path": str(path.resolve()),
                    "error": str(error),
                }
            )

    return records, unreadable_rows


def _write_inventory(
    records: list[ImageRecord],
    unreadable_rows: list[dict[str, str]],
) -> None:
    rows: list[dict[str, object]] = []

    for record in records:
        rows.append(
            {
                "class_name": record.class_name,
                "relative_path": record.relative_path,
                "path": str(record.path),
                "readable": True,
                "error": "",
                "sha256": record.sha256,
                "dhash_hex": f"{record.dhash:016x}",
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
            }
        )

    for row in unreadable_rows:
        rows.append(
            {
                "class_name": row["class_name"],
                "relative_path": row["relative_path"],
                "path": row["path"],
                "readable": False,
                "error": row["error"],
                "sha256": "",
                "dhash_hex": "",
                "width": "",
                "height": "",
                "mode": "",
            }
        )

    fieldnames = [
        "class_name",
        "relative_path",
        "path",
        "readable",
        "error",
        "sha256",
        "dhash_hex",
        "width",
        "height",
        "mode",
    ]

    with IMAGE_INVENTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_exact_duplicate_report(
    exact_groups: dict[str, list[ImageRecord]],
) -> tuple[int, int, int, int]:
    duplicate_group_count = 0
    duplicate_file_count = 0
    redundant_copy_count = 0
    cross_class_group_count = 0

    with EXACT_DUPLICATES_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "exact_group_id",
                "sha256",
                "class_name",
                "relative_path",
                "path",
                "group_size",
                "cross_class_duplicate",
            ]
        )

        for group_number, (
            digest,
            group,
        ) in enumerate(
            sorted(
                (
                    (digest, members)
                    for digest, members in exact_groups.items()
                    if len(members) >= 2
                ),
                key=lambda item: min(
                    member.relative_path for member in item[1]
                ),
            ),
            start=1,
        ):
            duplicate_group_count += 1
            duplicate_file_count += len(group)
            redundant_copy_count += len(group) - 1

            cross_class = len(
                {member.class_name for member in group}
            ) > 1

            if cross_class:
                cross_class_group_count += 1

            exact_group_id = f"EXACT_{group_number:04d}"

            for member in sorted(
                group,
                key=lambda item: item.relative_path,
            ):
                writer.writerow(
                    [
                        exact_group_id,
                        digest,
                        member.class_name,
                        member.relative_path,
                        str(member.path),
                        len(group),
                        cross_class,
                    ]
                )

    return (
        duplicate_group_count,
        duplicate_file_count,
        redundant_copy_count,
        cross_class_group_count,
    )


def _build_near_duplicate_rows(
    records: list[ImageRecord],
    threshold: int,
    previous_decisions: dict[tuple[str, str], dict[str, str]],
    auto_group_near: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    pair_number = 0

    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first.sha256 == second.sha256:
                continue

            distance = hamming_distance(first.dhash, second.dhash)

            if distance > threshold:
                continue

            pair_number += 1
            key = _pair_key(first.relative_path, second.relative_path)
            previous = previous_decisions.get(key, {})
            decision = _normalise_decision(previous.get("decision", ""))
            notes = str(previous.get("notes", ""))
            cross_class = first.class_name != second.class_name

            if auto_group_near and not decision and not cross_class:
                decision = "yes"
                if not notes:
                    notes = (
                        "Automatically grouped conservatively because "
                        f"dHash distance <= {threshold}."
                    )

            rows.append(
                {
                    "pair_id": f"NEAR_{pair_number:05d}",
                    "relative_path_1": first.relative_path,
                    "class_1": first.class_name,
                    "path_1": str(first.path),
                    "relative_path_2": second.relative_path,
                    "class_2": second.class_name,
                    "path_2": str(second.path),
                    "dhash_distance": distance,
                    "cross_class_candidate": cross_class,
                    "decision": decision,
                    "notes": notes,
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["dhash_distance"]),
            str(row["relative_path_1"]),
            str(row["relative_path_2"]),
        )
    )

    for index, row in enumerate(rows, start=1):
        row["pair_id"] = f"NEAR_{index:05d}"

    return rows


def _write_near_review(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair_id",
        "relative_path_1",
        "class_1",
        "path_1",
        "relative_path_2",
        "class_2",
        "path_2",
        "dhash_distance",
        "cross_class_candidate",
        "decision",
        "notes",
    ]

    with NEAR_DUPLICATE_REVIEW_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_group_manifest(
    records: list[ImageRecord],
    exact_groups: dict[str, list[ImageRecord]],
    near_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    record_by_relative = {
        record.relative_path: record
        for record in records
    }

    union_find = UnionFind(list(record_by_relative))

    exact_edge_members: set[str] = set()
    reviewed_near_edges: list[tuple[str, str]] = []

    for group in exact_groups.values():
        if len(group) < 2:
            continue

        class_names = {member.class_name for member in group}

        if len(class_names) > 1:
            continue

        first = group[0].relative_path

        for member in group[1:]:
            union_find.union(first, member.relative_path)

        exact_edge_members.update(
            member.relative_path for member in group
        )

    confirmed_cross_class_near = 0

    for row in near_rows:
        decision = _normalise_decision(row["decision"])

        if decision != "yes":
            continue

        first_relative = str(row["relative_path_1"])
        second_relative = str(row["relative_path_2"])
        first_record = record_by_relative[first_relative]
        second_record = record_by_relative[second_relative]

        if first_record.class_name != second_record.class_name:
            confirmed_cross_class_near += 1
            continue

        union_find.union(first_relative, second_relative)
        reviewed_near_edges.append((first_relative, second_relative))

    components: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        root = union_find.find(record.relative_path)
        components[root].append(record)

    near_members: set[str] = set()

    for first_relative, second_relative in reviewed_near_edges:
        near_members.add(first_relative)
        near_members.add(second_relative)

    components_by_class: dict[str, list[list[ImageRecord]]] = defaultdict(list)

    for members in components.values():
        class_names = {member.class_name for member in members}

        if len(class_names) != 1:
            raise RuntimeError(
                "A final duplicate group contains multiple classes. "
                "Resolve the label conflict before splitting."
            )

        class_name = next(iter(class_names))
        components_by_class[class_name].append(members)

    rows: list[dict[str, object]] = []

    for class_name in CLASS_NAMES:
        class_components = sorted(
            components_by_class.get(class_name, []),
            key=lambda members: min(
                member.relative_path for member in members
            ),
        )

        for group_number, members in enumerate(
            class_components,
            start=1,
        ):
            group_id = f"{class_name}_G{group_number:04d}"
            group_size = len(members)
            component_paths = {
                member.relative_path for member in members
            }
            contains_exact = bool(
                component_paths & exact_edge_members
            )
            contains_reviewed_near = bool(
                component_paths & near_members
            )

            if group_size == 1:
                grouping_basis = "singleton"
            elif contains_exact and contains_reviewed_near:
                grouping_basis = "exact_and_confirmed_near"
            elif contains_exact:
                grouping_basis = "exact"
            elif contains_reviewed_near:
                grouping_basis = "confirmed_near"
            else:
                grouping_basis = "linked_group"

            for member in sorted(
                members,
                key=lambda item: item.relative_path,
            ):
                rows.append(
                    {
                        "group_id": group_id,
                        "class_name": member.class_name,
                        "relative_path": member.relative_path,
                        "source_path": str(member.path),
                        "sha256": member.sha256,
                        "dhash_hex": f"{member.dhash:016x}",
                        "width": member.width,
                        "height": member.height,
                        "mode": member.mode,
                        "group_size": group_size,
                        "grouping_basis": grouping_basis,
                    }
                )

    fieldnames = [
        "group_id",
        "class_name",
        "relative_path",
        "source_path",
        "sha256",
        "dhash_hex",
        "width",
        "height",
        "mode",
        "group_size",
        "grouping_basis",
    ]

    with DATASET_GROUPS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, confirmed_cross_class_near


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the raw AniMedika dataset, preserve reviewed "
            "near-duplicate decisions, and build duplicate groups."
        )
    )

    parser.add_argument(
        "--near-threshold",
        type=int,
        default=NEAR_DUPLICATE_THRESHOLD,
        help=(
            "Maximum dHash Hamming distance used to flag potential "
            f"near-duplicates. Default: {NEAR_DUPLICATE_THRESHOLD}."
        ),
    )

    parser.add_argument(
        "--auto-group-near",
        action="store_true",
        help=(
            "Conservatively mark every unresolved same-class potential "
            "near-duplicate pair as yes. Existing yes/no decisions are kept."
        ),
    )

    args = parser.parse_args()

    if args.near_threshold < 0:
        raise ValueError("--near-threshold cannot be negative.")

    create_required_folders()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning raw dataset images...")
    records, unreadable_rows = _scan_images()
    _write_inventory(records, unreadable_rows)

    exact_groups: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        exact_groups[record.sha256].append(record)

    (
        exact_group_count,
        exact_file_count,
        redundant_exact_copy_count,
        cross_class_exact_group_count,
    ) = _write_exact_duplicate_report(exact_groups)

    previous_decisions = _load_existing_near_review()

    near_rows = _build_near_duplicate_rows(
        records=records,
        threshold=args.near_threshold,
        previous_decisions=previous_decisions,
        auto_group_near=args.auto_group_near,
    )

    _write_near_review(near_rows)

    group_rows, confirmed_cross_class_near = _build_group_manifest(
        records=records,
        exact_groups=exact_groups,
        near_rows=near_rows,
    )

    decision_counts = Counter(
        _normalise_decision(row["decision"])
        for row in near_rows
    )

    unresolved_near_pairs = decision_counts[""]
    confirmed_near_pairs = decision_counts["yes"]
    rejected_near_pairs = decision_counts["no"]
    cross_class_near_candidates = sum(
        bool(row["cross_class_candidate"])
        for row in near_rows
    )

    group_sizes: dict[str, int] = {}
    group_classes: dict[str, str] = {}

    for row in group_rows:
        group_id = str(row["group_id"])
        group_sizes[group_id] = int(row["group_size"])
        group_classes[group_id] = str(row["class_name"])

    class_file_counts = Counter(record.class_name for record in records)
    class_group_counts = Counter(group_classes.values())

    summary = {
        "raw_dataset_directory": str(RAW_DATASET_DIR),
        "near_duplicate_threshold": int(args.near_threshold),
        "auto_group_near_used": bool(args.auto_group_near),
        "supported_image_files_found": len(records) + len(unreadable_rows),
        "readable_image_files": len(records),
        "unreadable_image_files": len(unreadable_rows),
        "exact_duplicate_groups": exact_group_count,
        "files_in_exact_duplicate_groups": exact_file_count,
        "redundant_exact_copies": redundant_exact_copy_count,
        "cross_class_exact_duplicate_groups": cross_class_exact_group_count,
        "potential_near_duplicate_pairs": len(near_rows),
        "confirmed_near_duplicate_pairs": confirmed_near_pairs,
        "rejected_near_duplicate_pairs": rejected_near_pairs,
        "unresolved_near_duplicate_pairs": unresolved_near_pairs,
        "cross_class_near_duplicate_candidates": cross_class_near_candidates,
        "confirmed_cross_class_near_pairs": confirmed_cross_class_near,
        "final_duplicate_groups": len(group_sizes),
        "non_singleton_groups": sum(
            size > 1 for size in group_sizes.values()
        ),
        "largest_group_size": max(group_sizes.values(), default=0),
        "class_file_counts": {
            class_name: int(class_file_counts[class_name])
            for class_name in CLASS_NAMES
        },
        "class_group_counts": {
            class_name: int(class_group_counts[class_name])
            for class_name in CLASS_NAMES
        },
        "output_files": {
            "image_inventory": str(IMAGE_INVENTORY_PATH),
            "exact_duplicates": str(EXACT_DUPLICATES_PATH),
            "near_duplicate_review": str(NEAR_DUPLICATE_REVIEW_PATH),
            "dataset_groups": str(DATASET_GROUPS_PATH),
        },
    }

    AUDIT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nDataset audit complete.")
    print(f"Readable image files: {len(records)}")
    print(f"Unreadable image files: {len(unreadable_rows)}")
    print(f"Exact duplicate groups: {exact_group_count}")
    print(f"Redundant exact copies: {redundant_exact_copy_count}")
    print(f"Potential near-duplicate pairs: {len(near_rows)}")
    print(f"Confirmed near pairs: {confirmed_near_pairs}")
    print(f"Rejected near pairs: {rejected_near_pairs}")
    print(f"Unresolved near pairs: {unresolved_near_pairs}")
    print(f"Final duplicate/source groups: {len(group_sizes)}")
    print(f"Largest group size: {max(group_sizes.values(), default=0)}")

    print("\nGenerated reports:")
    print(IMAGE_INVENTORY_PATH)
    print(EXACT_DUPLICATES_PATH)
    print(NEAR_DUPLICATE_REVIEW_PATH)
    print(DATASET_GROUPS_PATH)
    print(AUDIT_SUMMARY_PATH)

    blocking_problems: list[str] = []

    if unreadable_rows:
        blocking_problems.append(
            "Unreadable images were detected. Remove or replace them."
        )

    if cross_class_exact_group_count:
        blocking_problems.append(
            "Exact duplicate groups span different classes. Correct the labels."
        )

    if confirmed_cross_class_near:
        blocking_problems.append(
            "A confirmed near-duplicate pair spans different classes. "
            "Change its decision or correct the labels."
        )

    if unresolved_near_pairs:
        blocking_problems.append(
            "Potential near-duplicate pairs still need yes/no decisions. "
            "Open near_duplicate_review.csv, fill the decision column, "
            "and rerun this script. Alternatively, rerun with "
            "--auto-group-near for conservative same-class grouping."
        )

    if blocking_problems:
        print("\nDo not create the permanent split yet:")
        for problem in blocking_problems:
            print(f"- {problem}")
    else:
        print(
            "\nAudit is ready for group-aware splitting. "
            "Run: py -3.13 src\\split_dataset.py"
        )


if __name__ == "__main__":
    main()