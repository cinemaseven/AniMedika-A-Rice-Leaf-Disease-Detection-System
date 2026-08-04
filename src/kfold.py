from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASS_NAMES, N_SPLITS, create_required_folders
from dataset import (
    create_dataset_from_paths,
    get_split_paths_labels_groups,
)
from experiments import get_experiment, save_experiment_config
from metrics import (
    calculate_metrics,
    collect_predictions,
    compare_metric_sets,
    save_classification_report,
    save_confusion_matrix,
    save_json,
    save_predictions,
    save_roc_curve,
)
from paths import get_experiment_paths
from training_utils import train_two_phase


def summarize_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Calculate descriptive statistics for selected fold-result columns."""

    rows: list[dict] = []

    # Critical t-value for a 95% confidence interval with five folds:
    # df = 5 - 1 = 4.
    t_critical_df4 = 2.776

    for column in columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        mean = values.mean()
        std = values.std(ddof=1)

        if len(values) > 1:
            margin = (
                t_critical_df4
                * std
                / math.sqrt(len(values))
            )
        else:
            margin = 0.0

        rows.append(
            {
                "metric": column,
                "mean": float(mean),
                "standard_deviation": (
                    float(std)
                    if not np.isnan(std)
                    else 0.0
                ),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "ci95_lower": float(mean - margin),
                "ci95_upper": float(mean + margin),
                "fold_count": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


def enable_tensorflow_determinism(seed: int) -> None:
    """
    Configure Python, NumPy, and TensorFlow randomness through
    TensorFlow's unified random-seed function.
    """

    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("TensorFlow deterministic operations enabled.")
    except (AttributeError, RuntimeError) as error:
        print(
            "Warning: TensorFlow deterministic operations "
            f"could not be enabled: {error}"
        )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        class_name: int(np.sum(labels == class_index))
        for class_index, class_name in enumerate(CLASS_NAMES)
    }


def _verify_fold_groups(
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    fold: int,
) -> tuple[int, int, int]:
    train_group_set = set(train_groups.tolist())
    validation_group_set = set(validation_groups.tolist())
    overlap = train_group_set & validation_group_set

    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise RuntimeError(
            f"Group leakage detected in fold {fold}. "
            f"Overlapping groups include: {preview}"
        )

    return (
        len(train_group_set),
        len(validation_group_set),
        len(overlap),
    )


def combine_validation_classification_reports(
    kfold_dir: Path,
    experiment_name: str,
) -> Path:
    """Combine all fold validation reports into one readable text file."""

    report_paths = list(
        kfold_dir.rglob("validation_classification_report.txt")
    )

    def fold_sort_key(report_path: Path) -> tuple[str, int]:
        fold_folder = report_path.parent.name

        try:
            fold_number = int(fold_folder.replace("fold_", ""))
        except ValueError:
            fold_number = 999

        return str(report_path.parent.parent), fold_number

    report_paths.sort(key=fold_sort_key)

    if not report_paths:
        raise FileNotFoundError(
            f"No validation classification reports found in {kfold_dir}"
        )

    output_path = (
        kfold_dir.parent
        / f"{experiment_name}_classification_reports.txt"
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            f"{experiment_name.upper()} "
            "5-FOLD VALIDATION CLASSIFICATION REPORTS\n"
        )
        output_file.write("=" * 70 + "\n")

        for report_path in report_paths:
            fold_name = report_path.parent.name.upper()

            output_file.write("\n")
            output_file.write("=" * 70 + "\n")
            output_file.write(f"{fold_name}\n")
            output_file.write("=" * 70 + "\n\n")

            report_text = report_path.read_text(encoding="utf-8")
            output_file.write(report_text.rstrip())
            output_file.write("\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run duplicate-group-aware five-fold cross-validation "
            "for an AniMedika experiment."
        )
    )

    parser.add_argument(
        "--experiment",
        default="baseline",
        help="Experiment name defined in experiments.py.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed. When supplied, results are saved "
            "in a seed-specific experiment folder."
        ),
    )

    args = parser.parse_args()

    create_required_folders()

    experiment = get_experiment(args.experiment)

    if args.seed is not None:
        experiment["seed"] = int(args.seed)

    base_seed = int(experiment["seed"])

    if args.seed is None:
        run_name = args.experiment
    else:
        run_name = f"{args.experiment}_seed{base_seed}"

    experiment["base_experiment"] = args.experiment
    experiment["run_name"] = run_name
    experiment["cross_validation_splitter"] = "StratifiedGroupKFold"
    experiment["group_source"] = "dataset/split/split_manifest.csv"

    paths = get_experiment_paths(run_name)

    save_experiment_config(
        experiment,
        paths.kfold_dir / "experiment_config.json",
    )

    enable_tensorflow_determinism(base_seed)

    print("\nCross-validation configuration")
    print(f"Base experiment: {args.experiment}")
    print(f"Run name:       {run_name}")
    print(f"Random seed:    {base_seed}")
    print(f"Number of folds: {N_SPLITS}")
    print("Splitter:       StratifiedGroupKFold")

    image_paths, labels, groups = get_split_paths_labels_groups("train")

    image_paths_array = np.asarray(image_paths)
    labels_array = np.asarray(labels, dtype=int)
    groups_array = np.asarray(groups)

    if len(image_paths_array) != len(labels_array) or len(labels_array) != len(groups_array):
        raise RuntimeError(
            "Training paths, labels, and groups do not have equal lengths."
        )

    group_to_classes: dict[str, set[int]] = {}

    for group_id, label in zip(groups_array, labels_array, strict=True):
        group_to_classes.setdefault(str(group_id), set()).add(int(label))

    cross_class_groups = {
        group_id: class_indices
        for group_id, class_indices in group_to_classes.items()
        if len(class_indices) > 1
    }

    if cross_class_groups:
        raise RuntimeError(
            "At least one duplicate/source group contains more than one "
            "class label. Correct the audit before cross-validation."
        )

    groups_per_class = {
        class_name: len(
            {
                str(group_id)
                for group_id, label in zip(
                    groups_array,
                    labels_array,
                    strict=True,
                )
                if int(label) == class_index
            }
        )
        for class_index, class_name in enumerate(CLASS_NAMES)
    }

    insufficient_group_classes = {
        class_name: count
        for class_name, count in groups_per_class.items()
        if count < N_SPLITS
    }

    if insufficient_group_classes:
        details = ", ".join(
            f"{class_name}={count}"
            for class_name, count in insufficient_group_classes.items()
        )
        raise RuntimeError(
            "Each class must contain at least one independent group per fold. "
            f"Insufficient group counts: {details}"
        )

    print(f"Training image files: {len(image_paths_array)}")
    print(f"Training groups:      {len(set(groups_array.tolist()))}")
    print("Groups per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {groups_per_class[class_name]}")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=base_seed,
    )

    fold_rows: list[dict] = []
    per_class_rows: list[pd.DataFrame] = []
    group_audit_rows: list[dict] = []
    fold_assignment_rows: list[dict] = []

    oof_true: list[int] = []
    oof_pred: list[int] = []
    oof_probs: list[np.ndarray] = []
    oof_paths: list[str] = []
    oof_groups: list[str] = []
    oof_folds: list[int] = []
    oof_seeds: list[int] = []

    split_iterator = splitter.split(
        image_paths_array,
        labels_array,
        groups_array,
    )

    for fold, (
        train_indices,
        validation_indices,
    ) in enumerate(split_iterator, start=1):
        print(
            f"\n========== RUN {run_name} "
            f"| SEED {base_seed} "
            f"| FOLD {fold} =========="
        )

        tf.keras.backend.clear_session()

        fold_seed = base_seed + fold
        tf.keras.utils.set_random_seed(fold_seed)
        print(f"Fold seed: {fold_seed}")

        fold_dir = paths.kfold_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_paths = image_paths_array[train_indices].tolist()
        train_labels = labels_array[train_indices].tolist()
        train_groups = groups_array[train_indices]

        validation_paths = image_paths_array[validation_indices].tolist()
        validation_labels = labels_array[validation_indices].tolist()
        validation_groups = groups_array[validation_indices]

        (
            train_group_count,
            validation_group_count,
            overlap_group_count,
        ) = _verify_fold_groups(
            train_groups,
            validation_groups,
            fold,
        )

        train_class_counts = _class_counts(
            np.asarray(train_labels, dtype=int)
        )
        validation_class_counts = _class_counts(
            np.asarray(validation_labels, dtype=int)
        )

        print(f"Fold-training images:   {len(train_paths)}")
        print(f"Fold-validation images: {len(validation_paths)}")
        print(f"Fold-training groups:   {train_group_count}")
        print(f"Fold-validation groups: {validation_group_count}")
        print("Group overlap:           0")

        group_audit_row: dict[str, object] = {
            "fold": fold,
            "seed": base_seed,
            "fold_seed": fold_seed,
            "train_image_count": len(train_paths),
            "validation_image_count": len(validation_paths),
            "train_group_count": train_group_count,
            "validation_group_count": validation_group_count,
            "overlap_group_count": overlap_group_count,
        }

        for class_name in CLASS_NAMES:
            group_audit_row[
                f"train_{class_name}_count"
            ] = train_class_counts[class_name]
            group_audit_row[
                f"validation_{class_name}_count"
            ] = validation_class_counts[class_name]

        group_audit_rows.append(group_audit_row)

        for index in train_indices:
            fold_assignment_rows.append(
                {
                    "fold": fold,
                    "role": "train",
                    "image_path": image_paths_array[index],
                    "class_name": CLASS_NAMES[int(labels_array[index])],
                    "class_index": int(labels_array[index]),
                    "group_id": groups_array[index],
                    "seed": base_seed,
                    "fold_seed": fold_seed,
                }
            )

        for index in validation_indices:
            fold_assignment_rows.append(
                {
                    "fold": fold,
                    "role": "validation",
                    "image_path": image_paths_array[index],
                    "class_name": CLASS_NAMES[int(labels_array[index])],
                    "class_index": int(labels_array[index]),
                    "group_id": groups_array[index],
                    "seed": base_seed,
                    "fold_seed": fold_seed,
                }
            )

        train_dataset = create_dataset_from_paths(
            train_paths,
            train_labels,
            shuffle=True,
            seed=fold_seed,
        )

        train_evaluation_dataset = create_dataset_from_paths(
            train_paths,
            train_labels,
            shuffle=False,
            seed=fold_seed,
        )

        validation_dataset = create_dataset_from_paths(
            validation_paths,
            validation_labels,
            shuffle=False,
            seed=fold_seed,
        )

        model, _, selection = train_two_phase(
            experiment,
            train_dataset,
            validation_dataset,
            fold_dir,
        )

        train_true, train_pred, train_probs = collect_predictions(
            model,
            train_evaluation_dataset,
        )

        (
            validation_true,
            validation_pred,
            validation_probs,
        ) = collect_predictions(
            model,
            validation_dataset,
        )

        train_metrics, train_per_class = calculate_metrics(
            train_true,
            train_pred,
            train_probs,
        )

        validation_metrics, validation_per_class = calculate_metrics(
            validation_true,
            validation_pred,
            validation_probs,
        )

        save_json(
            train_metrics,
            fold_dir / "train_metrics.json",
        )

        save_json(
            validation_metrics,
            fold_dir / "validation_metrics.json",
        )

        train_per_class.to_csv(
            fold_dir / "train_per_class_metrics.csv",
            index=False,
        )

        validation_per_class.to_csv(
            fold_dir / "validation_per_class_metrics.csv",
            index=False,
        )

        save_confusion_matrix(
            train_true,
            train_pred,
            fold_dir / "train_confusion_matrix.png",
            f"Fold {fold} Train",
        )

        save_confusion_matrix(
            validation_true,
            validation_pred,
            fold_dir / "validation_confusion_matrix.png",
            f"Fold {fold} Validation",
        )

        save_classification_report(
            validation_true,
            validation_pred,
            fold_dir / "validation_classification_report.txt",
        )

        save_roc_curve(
            validation_true,
            validation_probs,
            fold_dir / "validation_roc_curve.png",
            f"Fold {fold} Validation ROC",
        )

        save_predictions(
            validation_true,
            validation_pred,
            validation_probs,
            fold_dir / "validation_predictions.csv",
            image_paths=validation_paths,
            extra_columns={
                "group_id": validation_groups.tolist(),
                "fold": [fold] * len(validation_true),
                "seed": [base_seed] * len(validation_true),
                "fold_seed": [fold_seed] * len(validation_true),
            },
        )

        gaps = compare_metric_sets(
            "train",
            train_metrics,
            "validation",
            validation_metrics,
        )

        gaps.to_csv(
            fold_dir / "train_validation_gaps.csv",
            index=False,
        )

        row: dict[str, object] = {
            "fold": fold,
            "seed": base_seed,
            "fold_seed": fold_seed,
            "train_image_count": len(train_paths),
            "validation_image_count": len(validation_paths),
            "train_group_count": train_group_count,
            "validation_group_count": validation_group_count,
            "overlap_group_count": overlap_group_count,
            "selected_phase": selection["selected_phase"],
        }

        for key, value in train_metrics.items():
            row[f"train_{key}"] = value

        for key, value in validation_metrics.items():
            row[f"validation_{key}"] = value

        for _, gap_row in gaps.iterrows():
            metric_name = gap_row["metric"]
            row[f"gap_{metric_name}"] = gap_row[
                "generalization_gap"
            ]

        fold_rows.append(row)

        validation_per_class.insert(0, "seed", base_seed)
        validation_per_class.insert(0, "fold", fold)
        per_class_rows.append(validation_per_class)

        oof_true.extend(validation_true.tolist())
        oof_pred.extend(validation_pred.tolist())
        oof_probs.extend(validation_probs)
        oof_paths.extend(validation_paths)
        oof_groups.extend(validation_groups.tolist())
        oof_folds.extend([fold] * len(validation_true))
        oof_seeds.extend([base_seed] * len(validation_true))

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(
        paths.kfold_dir / "kfold_fold_results.csv",
        index=False,
    )

    group_audit_frame = pd.DataFrame(group_audit_rows)
    group_audit_frame.to_csv(
        paths.kfold_dir / "kfold_group_integrity.csv",
        index=False,
    )

    fold_assignment_frame = pd.DataFrame(fold_assignment_rows)
    fold_assignment_frame.to_csv(
        paths.kfold_dir / "kfold_group_assignments.csv",
        index=False,
    )

    summary_columns = [
        column
        for column in fold_frame.columns
        if (
            column.startswith("validation_")
            or column.startswith("train_")
            or column.startswith("gap_")
        )
    ]

    summary = summarize_columns(fold_frame, summary_columns)
    summary.to_csv(
        paths.kfold_dir / "kfold_summary.csv",
        index=False,
    )

    summary_json = {
        row["metric"]: {
            key: value
            for key, value in row.items()
            if key != "metric"
        }
        for row in summary.to_dict("records")
    }

    save_json(
        summary_json,
        paths.kfold_dir / "kfold_summary.json",
    )

    per_class_frame = pd.concat(
        per_class_rows,
        ignore_index=True,
    )

    per_class_frame.to_csv(
        paths.kfold_dir / "kfold_per_class_results.csv",
        index=False,
    )

    class_summary = (
        per_class_frame
        .groupby("class")
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall_sensitivity", "mean"),
            recall_std=("recall_sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
            support_total=("support", "sum"),
        )
        .reset_index()
    )

    class_summary.insert(0, "seed", base_seed)
    class_summary.to_csv(
        paths.kfold_dir / "kfold_per_class_summary.csv",
        index=False,
    )

    oof_true_array = np.asarray(oof_true)
    oof_pred_array = np.asarray(oof_pred)
    oof_probs_array = np.asarray(oof_probs)

    if len(oof_paths) != len(image_paths_array):
        raise RuntimeError(
            "OOF prediction count does not equal the number of training images."
        )

    if len(oof_paths) != len(set(oof_paths)):
        raise RuntimeError(
            "At least one training image appeared in more than one held-out fold."
        )

    expected_paths = set(image_paths_array.tolist())
    observed_paths = set(oof_paths)

    if expected_paths != observed_paths:
        missing = expected_paths - observed_paths
        extra = observed_paths - expected_paths
        raise RuntimeError(
            "OOF coverage mismatch. "
            f"Missing={len(missing)}, extra={len(extra)}."
        )

    oof_metrics, oof_per_class = calculate_metrics(
        oof_true_array,
        oof_pred_array,
        oof_probs_array,
    )

    oof_metrics["seed"] = base_seed
    oof_metrics["run_name"] = run_name
    oof_metrics["base_experiment"] = args.experiment
    oof_metrics["cross_validation_splitter"] = "StratifiedGroupKFold"
    oof_metrics["group_overlap_detected"] = False
    oof_metrics["unique_group_count"] = len(set(groups_array.tolist()))

    save_json(
        oof_metrics,
        paths.kfold_dir / "oof_metrics.json",
    )

    oof_per_class.insert(0, "seed", base_seed)
    oof_per_class.to_csv(
        paths.kfold_dir / "oof_per_class_metrics.csv",
        index=False,
    )

    save_confusion_matrix(
        oof_true_array,
        oof_pred_array,
        paths.kfold_dir / "oof_confusion_matrix.png",
        (
            "Aggregated Group-Aware Out-of-Fold "
            f"Confusion Matrix — Seed {base_seed}"
        ),
    )

    save_roc_curve(
        oof_true_array,
        oof_probs_array,
        paths.kfold_dir / "oof_roc_curve.png",
        (
            "Aggregated Group-Aware Out-of-Fold "
            f"ROC Curves — Seed {base_seed}"
        ),
    )

    save_predictions(
        oof_true_array,
        oof_pred_array,
        oof_probs_array,
        paths.kfold_dir / "oof_predictions.csv",
        image_paths=oof_paths,
        extra_columns={
            "group_id": oof_groups,
            "fold": oof_folds,
            "seed": oof_seeds,
        },
    )

    combined_report_path = combine_validation_classification_reports(
        paths.kfold_dir,
        run_name,
    )

    print("\nGroup-aware k-fold cross-validation complete.")
    print(f"Experiment: {args.experiment}")
    print(f"Run name:   {run_name}")
    print(f"Seed:       {base_seed}")
    print("Verified: no duplicate/source group crossed a fold boundary.")
    print(f"Combined classification report: {combined_report_path}")

    print("\nSend these files for model review:")

    for filename in [
        "kfold_fold_results.csv",
        "kfold_summary.csv",
        "kfold_per_class_summary.csv",
        "oof_metrics.json",
        "oof_confusion_matrix.png",
        "oof_predictions.csv",
        "kfold_group_integrity.csv",
    ]:
        print(paths.kfold_dir / filename)

    print(combined_report_path)


if __name__ == "__main__":
    main()