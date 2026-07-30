from __future__ import annotations

import argparse
import json
import shutil
import tensorflow as tf

import pandas as pd

from config import CLASS_NAMES, create_required_folders
from dataset import get_split_dataset, save_labels
from experiments import get_experiment, save_experiment_config
from metrics import compare_metric_sets, save_json, save_metric_bundle
from paths import get_experiment_paths
from training_utils import train_two_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    args = parser.parse_args()

    create_required_folders()
    experiment = get_experiment(args.experiment)
    paths = get_experiment_paths(args.experiment)
    save_experiment_config(experiment, paths.training_dir / "experiment_config.json")
    save_labels(paths.labels_path)

    train_dataset, _ = get_split_dataset("train", shuffle=True, seed=experiment["seed"])
    train_eval_dataset, train_paths = get_split_dataset("train", shuffle=False, seed=experiment["seed"])
    validation_dataset, validation_paths = get_split_dataset("validation", shuffle=False, seed=experiment["seed"])

    model, history, selection = train_two_phase(
        experiment,
        train_dataset,
        validation_dataset,
        paths.training_dir,
    )

    training_best = paths.training_dir / "best_model.keras"
    shutil.copy2(training_best, paths.best_model_path)

    # Reload the checkpoint that actually won phase selection.
    model = tf.keras.models.load_model(paths.best_model_path)

    # Ensure final_model.keras and all reports use the same selected model.
    model.save(paths.final_model_path)

    train_metrics, train_per_class, *_ = save_metric_bundle(
        model,
        train_eval_dataset,
        paths.training_dir,
        "train",
        image_paths=train_paths,
    )
    validation_metrics, validation_per_class, *_ = save_metric_bundle(
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
    split_metrics.to_csv(paths.training_dir / "train_validation_metrics.csv", index=False)

    gaps = compare_metric_sets("train", train_metrics, "validation", validation_metrics)
    gaps.to_csv(paths.training_dir / "train_validation_gaps.csv", index=False)

    results = {
        "experiment": args.experiment,
        "selection": selection,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "best_model_path": str(paths.best_model_path),
        "final_model_path": str(paths.final_model_path),
        "epochs_completed": int(len(history)),
    }
    save_json(results, paths.training_dir / "experiment_results.json")

    print("\nFinal training complete.")
    print(f"Best model: {paths.best_model_path}")
    print(f"Training/validation comparison: {paths.training_dir / 'train_validation_metrics.csv'}")
    print(f"Training/validation gaps: {paths.training_dir / 'train_validation_gaps.csv'}")


if __name__ == "__main__":
    main()
