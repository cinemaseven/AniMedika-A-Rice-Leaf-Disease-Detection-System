from __future__ import annotations

import argparse
import json

import pandas as pd
import tensorflow as tf

from dataset import get_split_dataset
from metrics import compare_metric_sets, save_json, save_metric_bundle
from paths import get_experiment_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate training reports and final_model.keras "
            "from the selected best_model.keras checkpoint."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment whose selected-model reports will be regenerated.",
    )
    args = parser.parse_args()

    paths = get_experiment_paths(args.experiment)

    if not paths.best_model_path.exists():
        raise FileNotFoundError(
            f"Selected best model not found: {paths.best_model_path}"
        )

    print(f"Loading selected model: {paths.best_model_path}")

    # Load the actual phase-selected checkpoint.
    model = tf.keras.models.load_model(paths.best_model_path)

    # Replace final_model.keras with the selected checkpoint.
    model.save(paths.final_model_path)

    train_dataset, train_paths = get_split_dataset(
        "train",
        shuffle=False,
        seed=42,
    )

    validation_dataset, validation_paths = get_split_dataset(
        "validation",
        shuffle=False,
        seed=42,
    )

    train_metrics, _, *_ = save_metric_bundle(
        model,
        train_dataset,
        paths.training_dir,
        "train",
        image_paths=train_paths,
    )

    validation_metrics, _, *_ = save_metric_bundle(
        model,
        validation_dataset,
        paths.training_dir,
        "validation",
        image_paths=validation_paths,
    )

    split_metrics = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "validation", **validation_metrics},
        ]
    )

    split_metrics.to_csv(
        paths.training_dir / "train_validation_metrics.csv",
        index=False,
    )

    gaps = compare_metric_sets(
        "train",
        train_metrics,
        "validation",
        validation_metrics,
    )

    gaps.to_csv(
        paths.training_dir / "train_validation_gaps.csv",
        index=False,
    )

    results_path = paths.training_dir / "experiment_results.json"

    if results_path.exists():
        results = json.loads(
            results_path.read_text(encoding="utf-8")
        )
    else:
        results = {
            "experiment": args.experiment,
        }

    results["train_metrics"] = train_metrics
    results["validation_metrics"] = validation_metrics
    results["best_model_path"] = str(paths.best_model_path)
    results["final_model_path"] = str(paths.final_model_path)
    results["reports_regenerated_from_selected_checkpoint"] = True

    save_json(
        results,
        results_path,
    )

    print("\nSelected-model outputs repaired.")
    print(f"Best model:  {paths.best_model_path}")
    print(f"Final model: {paths.final_model_path}")
    print(
        "Metrics:     "
        f"{paths.training_dir / 'train_validation_metrics.csv'}"
    )
    print(
        "Gaps:        "
        f"{paths.training_dir / 'train_validation_gaps.csv'}"
    )


if __name__ == "__main__":
    main()