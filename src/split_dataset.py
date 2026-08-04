from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from config import (
    AUDIT_SUMMARY_PATH,
    CLASS_NAMES,
    DATASET_GROUPS_PATH,
    NEAR_DUPLICATE_REVIEW_PATH,
    RAW_DATASET_DIR,
    SEED,
    SPLIT_DATASET_DIR,
    SPLIT_MANIFEST_PATH,
    SPLIT_SUMMARY_PATH,
    TEST_DIR,
    TEST_RATIO,
    TRAIN_DIR,
    TRAIN_RATIO,
    VALIDATION_DIR,
    VALIDATION_RATIO,
    create_required_folders,
)


CLEAR_EXISTING_SPLIT = True


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    class_name: str
    members: tuple[dict[str, str], ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class AssignmentState:
    test_count: int
    validation_count: int
    assignments: tuple[tuple[str, str], ...]


def _safe_destination(
    source: Path,
    class_root: Path,
    destination_dir: Path,
) -> Path:
    relative = source.relative_to(class_root)
    flattened = "__".join(relative.parts)
    destination = destination_dir / flattened
    counter = 1

    while destination.exists():
        destination = (
            destination_dir
            / f"{source.stem}_{counter}{source.suffix.lower()}"
        )
        counter += 1

    return destination


def _normalise_review_decision(value: object) -> str:
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

    return aliases.get(decision, decision)


def _check_audit_ready(allow_unreviewed_near: bool) -> None:
    if not DATASET_GROUPS_PATH.exists():
        raise FileNotFoundError(
            "Missing dataset group manifest. Run "
            "py -3.13 src\\audit_dataset.py first."
        )

    if AUDIT_SUMMARY_PATH.exists():
        summary = json.loads(
            AUDIT_SUMMARY_PATH.read_text(encoding="utf-8")
        )

        unreadable = int(summary.get("unreadable_image_files", 0))
        cross_class_exact = int(
            summary.get("cross_class_exact_duplicate_groups", 0)
        )
        confirmed_cross_class_near = int(
            summary.get("confirmed_cross_class_near_pairs", 0)
        )

        if unreadable:
            raise RuntimeError(
                f"The audit found {unreadable} unreadable image files. "
                "Remove or replace them, then rerun the audit."
            )

        if cross_class_exact:
            raise RuntimeError(
                "The audit found exact duplicates assigned to different "
                "classes. Correct the labels before splitting."
            )

        if confirmed_cross_class_near:
            raise RuntimeError(
                "The audit contains a confirmed near-duplicate pair assigned "
                "to different classes. Correct the decision or labels."
            )

    if not NEAR_DUPLICATE_REVIEW_PATH.exists():
        return

    unresolved = 0

    with NEAR_DUPLICATE_REVIEW_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not _normalise_review_decision(row.get("decision", "")):
                unresolved += 1

    if unresolved and not allow_unreviewed_near:
        raise RuntimeError(
            f"There are {unresolved} unresolved potential near-duplicate "
            "pairs. Fill the decision column in "
            f"{NEAR_DUPLICATE_REVIEW_PATH.name} and rerun the audit, or "
            "rerun the audit with --auto-group-near. The split was not created."
        )

    if unresolved and allow_unreviewed_near:
        print(
            "Warning: proceeding with unresolved near-duplicate candidates. "
            "Only exact and confirmed near-duplicate relationships are grouped."
        )


def _load_group_records() -> dict[str, list[GroupRecord]]:
    rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)

    with DATASET_GROUPS_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "group_id",
            "class_name",
            "relative_path",
            "sha256",
            "grouping_basis",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "dataset_groups.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            group_id = str(row["group_id"]).strip()
            class_name = str(row["class_name"]).strip()
            relative_path = str(row["relative_path"]).strip()

            if not group_id or not class_name or not relative_path:
                raise ValueError(
                    "dataset_groups.csv contains an incomplete row."
                )

            if class_name not in CLASS_NAMES:
                raise ValueError(
                    f"Unexpected class '{class_name}' in dataset_groups.csv."
                )

            rows_by_group[group_id].append(dict(row))

    groups_by_class: dict[str, list[GroupRecord]] = defaultdict(list)
    seen_relative_paths: set[str] = set()

    for group_id, members in rows_by_group.items():
        class_names = {str(member["class_name"]) for member in members}

        if len(class_names) != 1:
            raise RuntimeError(
                f"Group {group_id} spans multiple classes: "
                + ", ".join(sorted(class_names))
            )

        class_name = next(iter(class_names))

        for member in members:
            relative_path = str(member["relative_path"])

            if relative_path in seen_relative_paths:
                raise RuntimeError(
                    f"Image appears more than once in group manifest: {relative_path}"
                )

            seen_relative_paths.add(relative_path)

            source = RAW_DATASET_DIR / Path(relative_path)

            if not source.exists():
                raise FileNotFoundError(
                    f"Raw image listed in the group manifest was not found: {source}"
                )

        groups_by_class[class_name].append(
            GroupRecord(
                group_id=group_id,
                class_name=class_name,
                members=tuple(
                    sorted(
                        members,
                        key=lambda row: str(row["relative_path"]),
                    )
                ),
            )
        )

    for class_name in CLASS_NAMES:
        if not groups_by_class[class_name]:
            raise ValueError(
                f"No grouped images were found for class {class_name}."
            )

    return groups_by_class


def _assign_groups_to_splits(
    groups: list[GroupRecord],
    target_test: int,
    target_validation: int,
    seed: int,
) -> dict[str, str]:
    """
    Assign complete groups to train, validation, or test.

    A two-dimensional dynamic program searches for exact test and validation
    image counts. If exact counts are impossible because of group sizes, the
    closest deterministic assignment is selected and reported.
    """

    if target_test < 1 or target_validation < 1:
        raise ValueError("Test and validation targets must both be positive.")

    if target_test + target_validation >= sum(group.size for group in groups):
        raise ValueError("Train split would be empty for this class.")

    rng = random.Random(seed)
    ordered_groups = list(groups)
    rng.shuffle(ordered_groups)

    # Prefer larger groups first after the seeded shuffle. Stable sorting keeps
    # the seeded order among groups of equal size.
    ordered_groups.sort(key=lambda group: group.size, reverse=True)

    largest_group = max(group.size for group in ordered_groups)
    maximum_test = target_test + largest_group
    maximum_validation = target_validation + largest_group

    # Each state stores the complete choices made so far. The number of states
    # remains small for the present dataset because only test and validation
    # counts are tracked.
    states: dict[tuple[int, int], tuple[tuple[str, str], ...]] = {
        (0, 0): tuple()
    }

    for group in ordered_groups:
        next_states: dict[
            tuple[int, int],
            tuple[tuple[str, str], ...],
        ] = {}

        for (test_count, validation_count), assignments in states.items():
            train_key = (test_count, validation_count)
            next_states.setdefault(
                train_key,
                assignments + ((group.group_id, "train"),),
            )

            new_test = test_count + group.size

            if new_test <= maximum_test:
                key = (new_test, validation_count)
                next_states.setdefault(
                    key,
                    assignments + ((group.group_id, "test"),),
                )

            new_validation = validation_count + group.size

            if new_validation <= maximum_validation:
                key = (test_count, new_validation)
                next_states.setdefault(
                    key,
                    assignments + ((group.group_id, "validation"),),
                )

        states = next_states

    exact_key = (target_test, target_validation)

    if exact_key in states:
        selected_assignments = states[exact_key]
    else:
        total_images = sum(group.size for group in ordered_groups)

        def score(item: tuple[tuple[int, int], tuple[tuple[str, str], ...]]) -> tuple:
            (test_count, validation_count), assignments = item
            train_count = total_images - test_count - validation_count

            if train_count <= 0 or test_count <= 0 or validation_count <= 0:
                return (float("inf"),)

            absolute_error = (
                abs(test_count - target_test)
                + abs(validation_count - target_validation)
            )
            maximum_error = max(
                abs(test_count - target_test),
                abs(validation_count - target_validation),
            )
            total_holdout_error = abs(
                (test_count + validation_count)
                - (target_test + target_validation)
            )
            assignment_signature = tuple(assignments)

            return (
                absolute_error,
                maximum_error,
                total_holdout_error,
                assignment_signature,
            )

        selected_key, selected_assignments = min(
            states.items(),
            key=score,
        )

        print(
            "Warning: exact group-preserving counts were not possible. "
            f"Selected test={selected_key[0]} and "
            f"validation={selected_key[1]} instead of "
            f"test={target_test}, validation={target_validation}."
        )

    assignment_map = dict(selected_assignments)

    # Groups left in the initial state only appear in explicit assignments
    # because every processed group is appended in all branches.
    missing_groups = {
        group.group_id for group in groups
    } - set(assignment_map)

    if missing_groups:
        raise RuntimeError(
            "Internal group assignment error. Missing groups: "
            + ", ".join(sorted(missing_groups))
        )

    return assignment_map


def _verify_group_integrity(manifest_rows: list[dict[str, object]]) -> None:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    classes_by_group: dict[str, set[str]] = defaultdict(set)
    splits_by_sha: dict[str, set[str]] = defaultdict(set)

    for row in manifest_rows:
        group_id = str(row["group_id"])
        split = str(row["split"])
        class_name = str(row["class_name"])
        digest = str(row["sha256"])

        splits_by_group[group_id].add(split)
        classes_by_group[group_id].add(class_name)

        if digest:
            splits_by_sha[digest].add(split)

    crossing_groups = {
        group_id: splits
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    }

    if crossing_groups:
        details = "; ".join(
            f"{group_id}: {sorted(splits)}"
            for group_id, splits in sorted(crossing_groups.items())
        )
        raise RuntimeError(
            "Group leakage was detected after splitting: " + details
        )

    cross_class_groups = {
        group_id: classes
        for group_id, classes in classes_by_group.items()
        if len(classes) > 1
    }

    if cross_class_groups:
        raise RuntimeError(
            "A duplicate group contains multiple class labels after splitting."
        )

    crossing_exact_hashes = {
        digest: splits
        for digest, splits in splits_by_sha.items()
        if len(splits) > 1
    }

    if crossing_exact_hashes:
        raise RuntimeError(
            "Exact duplicate leakage was detected after splitting."
        )


def split_dataset(allow_unreviewed_near: bool = False) -> None:
    if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1.0."
        )

    _check_audit_ready(allow_unreviewed_near=allow_unreviewed_near)
    groups_by_class = _load_group_records()

    if CLEAR_EXISTING_SPLIT and SPLIT_DATASET_DIR.exists():
        shutil.rmtree(SPLIT_DATASET_DIR)

    create_required_folders()

    for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    split_directories = {
        "train": TRAIN_DIR,
        "validation": VALIDATION_DIR,
        "test": TEST_DIR,
    }

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_counts = Counter()

    print("Creating fixed group-aware 70/10/20 split per class...\n")

    for class_index, class_name in enumerate(CLASS_NAMES):
        groups = groups_by_class[class_name]
        total_images = sum(group.size for group in groups)

        test_target = round(total_images * TEST_RATIO)
        validation_target = round(total_images * VALIDATION_RATIO)
        train_target = total_images - test_target - validation_target

        assignment_map = _assign_groups_to_splits(
            groups=groups,
            target_test=test_target,
            target_validation=validation_target,
            seed=SEED + class_index,
        )

        actual_counts = Counter()
        actual_group_counts = Counter()

        for group in sorted(groups, key=lambda item: item.group_id):
            split_name = assignment_map[group.group_id]
            destination_class_dir = (
                split_directories[split_name] / class_name
            )
            destination_class_dir.mkdir(parents=True, exist_ok=True)
            class_root = RAW_DATASET_DIR / class_name

            actual_group_counts[split_name] += 1

            for member in group.members:
                source = RAW_DATASET_DIR / Path(member["relative_path"])
                destination = _safe_destination(
                    source,
                    class_root,
                    destination_class_dir,
                )
                shutil.copy2(source, destination)

                destination_relative = destination.relative_to(
                    RAW_DATASET_DIR.parent
                ).as_posix()

                manifest_rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "class_index": class_index,
                        "group_id": group.group_id,
                        "group_size": group.size,
                        "grouping_basis": member.get(
                            "grouping_basis",
                            "",
                        ),
                        "source_relative_path": member["relative_path"],
                        "source_path": str(source.resolve()),
                        "destination_relative_path": destination_relative,
                        "destination_path": str(destination.resolve()),
                        "sha256": member.get("sha256", ""),
                        "dhash_hex": member.get("dhash_hex", ""),
                    }
                )

                actual_counts[split_name] += 1
                total_counts[split_name] += 1

        for split_name, target in [
            ("train", train_target),
            ("validation", validation_target),
            ("test", test_target),
        ]:
            summary_rows.append(
                {
                    "class_name": class_name,
                    "split": split_name,
                    "target_image_count": target,
                    "actual_image_count": actual_counts[split_name],
                    "difference_from_target": actual_counts[split_name] - target,
                    "group_count": actual_group_counts[split_name],
                    "total_class_images": total_images,
                    "total_class_groups": len(groups),
                }
            )

        print(
            f"{class_name}: "
            f"train={actual_counts['train']} "
            f"({actual_group_counts['train']} groups), "
            f"validation={actual_counts['validation']} "
            f"({actual_group_counts['validation']} groups), "
            f"test={actual_counts['test']} "
            f"({actual_group_counts['test']} groups)"
        )

    _verify_group_integrity(manifest_rows)

    manifest_fieldnames = [
        "split",
        "class_name",
        "class_index",
        "group_id",
        "group_size",
        "grouping_basis",
        "source_relative_path",
        "source_path",
        "destination_relative_path",
        "destination_path",
        "sha256",
        "dhash_hex",
    ]

    with SPLIT_MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_fieldnames = [
        "class_name",
        "split",
        "target_image_count",
        "actual_image_count",
        "difference_from_target",
        "group_count",
        "total_class_images",
        "total_class_groups",
    ]

    with SPLIT_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nGroup-aware split complete.")
    print(f"Training: {total_counts['train']}")
    print(f"Validation: {total_counts['validation']}")
    print(f"Testing: {total_counts['test']}")
    print(f"Total: {sum(total_counts.values())}")
    print(f"Split manifest: {SPLIT_MANIFEST_PATH}")
    print(f"Split summary: {SPLIT_SUMMARY_PATH}")
    print("Verified: no duplicate group or exact SHA-256 hash crosses splits.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a permanent group-aware 70/10/20 dataset split."
        )
    )

    parser.add_argument(
        "--allow-unreviewed-near",
        action="store_true",
        help=(
            "Proceed even when some potential near-duplicate pairs are "
            "unreviewed. This is not recommended for the final thesis run."
        ),
    )

    args = parser.parse_args()

    split_dataset(
        allow_unreviewed_near=args.allow_unreviewed_near,
    )


if __name__ == "__main__":
    main()