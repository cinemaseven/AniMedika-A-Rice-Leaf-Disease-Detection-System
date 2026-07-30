from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf

from dataset import get_split_dataset
from experiments import get_experiment
from metrics import compare_metric_sets, save_metric_bundle
from paths import get_experiment_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args()

    if not args.confirm_final:
        raise SystemExit("This script reads the test set. Re-run with --confirm-final.")

    experiment = get_experiment(args.experiment)
    paths = get_experiment_paths(args.experiment)
    comparison_dir = paths.evaluation_dir / "split_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(paths.best_model_path)

    metrics_by_split = {}
    for split_name in ["train", "validation", "test"]:
        dataset, image_paths = get_split_dataset(
            split_name, shuffle=False, seed=experiment["seed"]
        )
        metrics, *_ = save_metric_bundle(
            model,
            dataset,
            comparison_dir,
            split_name,
            image_paths=image_paths,
        )
        metrics_by_split[split_name] = metrics

    pd.DataFrame(
        [{"split": split_name, **metrics} for split_name, metrics in metrics_by_split.items()]
    ).to_csv(comparison_dir / "split_metrics.csv", index=False)

    gaps = pd.concat(
        [
            compare_metric_sets("train", metrics_by_split["train"], "validation", metrics_by_split["validation"]),
            compare_metric_sets("train", metrics_by_split["train"], "test", metrics_by_split["test"]),
            compare_metric_sets("validation", metrics_by_split["validation"], "test", metrics_by_split["test"]),
        ],
        ignore_index=True,
    )
    gaps.to_csv(comparison_dir / "pairwise_metric_gaps.csv", index=False)
    print(comparison_dir / "split_metrics.csv")
    print(comparison_dir / "pairwise_metric_gaps.csv")


if __name__ == "__main__":
    main()
