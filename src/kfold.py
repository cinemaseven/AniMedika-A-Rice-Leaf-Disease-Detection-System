from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from config import CLASS_NAMES, N_SPLITS, TRAIN_DIR, create_required_folders
from dataset import create_dataset_from_paths, get_image_paths_and_labels
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

    # Critical t-value for a 95% confidence interval with:
    # five folds - 1 = four degrees of freedom.
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

    Deterministic operations improve reproducibility, although exact
    equality can still depend on the TensorFlow version and hardware.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run five-fold cross-validation for an "
            "AniMedika experiment."
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

    # Load the selected experiment configuration.
    experiment = get_experiment(args.experiment)

    # Use the experiment's normal seed when --seed is not supplied.
    if args.seed is not None:
        experiment["seed"] = int(args.seed)

    base_seed = int(experiment["seed"])

    # Normal runs keep the original experiment folder.
    # Explicit multi-seed runs use separate folders.
    if args.seed is None:
        run_name = args.experiment
    else:
        run_name = (
            f"{args.experiment}_seed{base_seed}"
        )

    # Store both the original experiment name and the seeded run name.
    experiment["base_experiment"] = args.experiment
    experiment["run_name"] = run_name

    paths = get_experiment_paths(run_name)

    save_experiment_config(
        experiment,
        paths.kfold_dir / "experiment_config.json",
    )

    # Set the initial process-level seed and deterministic behavior.
    enable_tensorflow_determinism(base_seed)

    print("\nCross-validation configuration")
    print(f"Base experiment: {args.experiment}")
    print(f"Run name:       {run_name}")
    print(f"Random seed:    {base_seed}")
    print(f"Number of folds: {N_SPLITS}")

    image_paths, labels = get_image_paths_and_labels(
        TRAIN_DIR
    )

    image_paths_array = np.asarray(image_paths)
    labels_array = np.asarray(labels)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=base_seed,
    )

    fold_rows: list[dict] = []
    per_class_rows: list[pd.DataFrame] = []

    oof_true: list[int] = []
    oof_pred: list[int] = []
    oof_probs: list[np.ndarray] = []
    oof_paths: list[str] = []
    oof_folds: list[int] = []
    oof_seeds: list[int] = []

    for fold, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(
            image_paths_array,
            labels_array,
        ),
        start=1,
    ):
        print(
            f"\n========== RUN {run_name} "
            f"| SEED {base_seed} "
            f"| FOLD {fold} =========="
        )

        # Remove the previous fold's model from memory.
        tf.keras.backend.clear_session()

        # Each fold receives a reproducible but different seed.
        fold_seed = base_seed + fold
        tf.keras.utils.set_random_seed(fold_seed)

        print(f"Fold seed: {fold_seed}")

        fold_dir = (
            paths.kfold_dir
            / f"fold_{fold}"
        )
        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_paths = (
            image_paths_array[train_indices]
            .tolist()
        )
        train_labels = (
            labels_array[train_indices]
            .tolist()
        )

        validation_paths = (
            image_paths_array[validation_indices]
            .tolist()
        )
        validation_labels = (
            labels_array[validation_indices]
            .tolist()
        )

        print(
            f"Fold-training images:   "
            f"{len(train_paths)}"
        )
        print(
            f"Fold-validation images: "
            f"{len(validation_paths)}"
        )

        train_dataset = create_dataset_from_paths(
            train_paths,
            train_labels,
            shuffle=True,
            seed=fold_seed,
        )

        train_evaluation_dataset = (
            create_dataset_from_paths(
                train_paths,
                train_labels,
                shuffle=False,
                seed=fold_seed,
            )
        )

        validation_dataset = (
            create_dataset_from_paths(
                validation_paths,
                validation_labels,
                shuffle=False,
                seed=fold_seed,
            )
        )

        model, _, selection = train_two_phase(
            experiment,
            train_dataset,
            validation_dataset,
            fold_dir,
        )

        train_true, train_pred, train_probs = (
            collect_predictions(
                model,
                train_evaluation_dataset,
            )
        )

        (
            validation_true,
            validation_pred,
            validation_probs,
        ) = collect_predictions(
            model,
            validation_dataset,
        )

        train_metrics, train_per_class = (
            calculate_metrics(
                train_true,
                train_pred,
                train_probs,
            )
        )

        (
            validation_metrics,
            validation_per_class,
        ) = calculate_metrics(
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
            fold_dir
            / "train_per_class_metrics.csv",
            index=False,
        )

        validation_per_class.to_csv(
            fold_dir
            / "validation_per_class_metrics.csv",
            index=False,
        )

        save_confusion_matrix(
            train_true,
            train_pred,
            fold_dir
            / "train_confusion_matrix.png",
            f"Fold {fold} Train",
        )

        save_confusion_matrix(
            validation_true,
            validation_pred,
            fold_dir
            / "validation_confusion_matrix.png",
            f"Fold {fold} Validation",
        )

        save_classification_report(
            validation_true,
            validation_pred,
            fold_dir
            / "validation_classification_report.txt",
        )

        save_roc_curve(
            validation_true,
            validation_probs,
            fold_dir
            / "validation_roc_curve.png",
            f"Fold {fold} Validation ROC",
        )

        save_predictions(
            validation_true,
            validation_pred,
            validation_probs,
            fold_dir
            / "validation_predictions.csv",
            image_paths=validation_paths,
            extra_columns={
                "fold": [
                    fold
                ] * len(validation_true),
                "seed": [
                    base_seed
                ] * len(validation_true),
                "fold_seed": [
                    fold_seed
                ] * len(validation_true),
            },
        )

        gaps = compare_metric_sets(
            "train",
            train_metrics,
            "validation",
            validation_metrics,
        )

        gaps.to_csv(
            fold_dir
            / "train_validation_gaps.csv",
            index=False,
        )

        row: dict = {
            "fold": fold,
            "seed": base_seed,
            "fold_seed": fold_seed,
            "selected_phase": (
                selection["selected_phase"]
            ),
        }

        for key, value in train_metrics.items():
            row[f"train_{key}"] = value

        for key, value in validation_metrics.items():
            row[f"validation_{key}"] = value

        for _, gap_row in gaps.iterrows():
            metric_name = gap_row["metric"]
            row[f"gap_{metric_name}"] = (
                gap_row["generalization_gap"]
            )

        fold_rows.append(row)

        # Include the seed and fold in class-level results.
        validation_per_class.insert(
            0,
            "seed",
            base_seed,
        )
        validation_per_class.insert(
            0,
            "fold",
            fold,
        )

        per_class_rows.append(
            validation_per_class
        )

        oof_true.extend(
            validation_true.tolist()
        )
        oof_pred.extend(
            validation_pred.tolist()
        )
        oof_probs.extend(
            validation_probs
        )
        oof_paths.extend(
            validation_paths
        )
        oof_folds.extend(
            [fold] * len(validation_true)
        )
        oof_seeds.extend(
            [base_seed] * len(validation_true)
        )

    # ---------------------------------------------------------
    # Save fold-level results
    # ---------------------------------------------------------

    fold_frame = pd.DataFrame(
        fold_rows
    )

    fold_frame.to_csv(
        paths.kfold_dir
        / "kfold_fold_results.csv",
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

    summary = summarize_columns(
        fold_frame,
        summary_columns,
    )

    summary.to_csv(
        paths.kfold_dir
        / "kfold_summary.csv",
        index=False,
    )

    summary_json = {
        row["metric"]: {
            key: value
            for key, value in row.items()
            if key != "metric"
        }
        for row in summary.to_dict(
            "records"
        )
    }

    save_json(
        summary_json,
        paths.kfold_dir
        / "kfold_summary.json",
    )

    # ---------------------------------------------------------
    # Save per-class fold results
    # ---------------------------------------------------------

    per_class_frame = pd.concat(
        per_class_rows,
        ignore_index=True,
    )

    per_class_frame.to_csv(
        paths.kfold_dir
        / "kfold_per_class_results.csv",
        index=False,
    )

    class_summary = (
        per_class_frame
        .groupby("class")
        .agg(
            precision_mean=(
                "precision",
                "mean",
            ),
            precision_std=(
                "precision",
                "std",
            ),
            recall_mean=(
                "recall_sensitivity",
                "mean",
            ),
            recall_std=(
                "recall_sensitivity",
                "std",
            ),
            specificity_mean=(
                "specificity",
                "mean",
            ),
            specificity_std=(
                "specificity",
                "std",
            ),
            f1_mean=(
                "f1_score",
                "mean",
            ),
            f1_std=(
                "f1_score",
                "std",
            ),
            support_total=(
                "support",
                "sum",
            ),
        )
        .reset_index()
    )

    class_summary.insert(
        0,
        "seed",
        base_seed,
    )

    class_summary.to_csv(
        paths.kfold_dir
        / "kfold_per_class_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Calculate aggregated out-of-fold results
    # ---------------------------------------------------------

    oof_true_array = np.asarray(
        oof_true
    )
    oof_pred_array = np.asarray(
        oof_pred
    )
    oof_probs_array = np.asarray(
        oof_probs
    )

    (
        oof_metrics,
        oof_per_class,
    ) = calculate_metrics(
        oof_true_array,
        oof_pred_array,
        oof_probs_array,
    )

    # Add the seed to the JSON metrics.
    oof_metrics["seed"] = base_seed
    oof_metrics["run_name"] = run_name
    oof_metrics["base_experiment"] = (
        args.experiment
    )

    save_json(
        oof_metrics,
        paths.kfold_dir
        / "oof_metrics.json",
    )

    oof_per_class.insert(
        0,
        "seed",
        base_seed,
    )

    oof_per_class.to_csv(
        paths.kfold_dir
        / "oof_per_class_metrics.csv",
        index=False,
    )

    save_confusion_matrix(
        oof_true_array,
        oof_pred_array,
        paths.kfold_dir
        / "oof_confusion_matrix.png",
        (
            "Aggregated Out-of-Fold "
            f"Confusion Matrix — Seed {base_seed}"
        ),
    )

    save_roc_curve(
        oof_true_array,
        oof_probs_array,
        paths.kfold_dir
        / "oof_roc_curve.png",
        (
            "Aggregated Out-of-Fold "
            f"ROC Curves — Seed {base_seed}"
        ),
    )

    save_predictions(
        oof_true_array,
        oof_pred_array,
        oof_probs_array,
        paths.kfold_dir
        / "oof_predictions.csv",
        image_paths=oof_paths,
        extra_columns={
            "fold": oof_folds,
            "seed": oof_seeds,
        },
    )

    print(
        "\nK-fold cross-validation complete."
    )
    print(f"Experiment: {args.experiment}")
    print(f"Run name:   {run_name}")
    print(f"Seed:       {base_seed}")

    print(
        "\nSend these files for model review:"
    )

    for filename in [
        "kfold_fold_results.csv",
        "kfold_summary.csv",
        "kfold_per_class_summary.csv",
        "oof_metrics.json",
        "oof_confusion_matrix.png",
    ]:
        print(
            paths.kfold_dir
            / filename
        )


if __name__ == "__main__":
    main()