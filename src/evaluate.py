from __future__ import annotations

import argparse
import json

import pandas as pd
import tensorflow as tf

from calibration import apply_temperature
from dataset import get_split_dataset
from experiments import get_experiment, save_experiment_config
from metrics import calculate_metrics, collect_predictions, save_json, save_metric_bundle
from paths import get_experiment_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args()

    if not args.confirm_final:
        raise SystemExit(
            "Final test evaluation is locked. Re-run with --confirm-final only after settings are selected."
        )

    experiment = get_experiment(args.experiment)
    paths = get_experiment_paths(args.experiment)
    save_experiment_config(experiment, paths.evaluation_dir / "experiment_config.json")

    model = tf.keras.models.load_model(paths.best_model_path)
    test_dataset, test_paths = get_split_dataset("test", shuffle=False, seed=experiment["seed"])
    raw_metrics, _, y_true, y_pred, y_probs = save_metric_bundle(
        model,
        test_dataset,
        paths.evaluation_dir,
        "test_raw",
        image_paths=test_paths,
    )

    result = {"raw_test_metrics": raw_metrics}
    if paths.temperature_path.exists():
        temperature_data = json.loads(paths.temperature_path.read_text(encoding="utf-8"))
        temperature = float(temperature_data["temperature"])
        calibrated_probs = apply_temperature(y_probs, temperature)
        calibrated_pred = calibrated_probs.argmax(axis=1)
        calibrated_metrics, calibrated_per_class = calculate_metrics(
            y_true, calibrated_pred, calibrated_probs
        )
        calibrated_per_class.to_csv(
            paths.evaluation_dir / "test_calibrated_per_class_metrics.csv", index=False
        )
        comparison = pd.DataFrame(
            [
                {"probability_type": "raw", **raw_metrics},
                {"probability_type": "temperature_scaled", **calibrated_metrics},
            ]
        )
        comparison.to_csv(
            paths.evaluation_dir / "test_calibration_comparison.csv", index=False
        )
        result["temperature"] = temperature
        result["calibrated_test_metrics"] = calibrated_metrics

    save_json(result, paths.evaluation_dir / "final_test_results.json")
    print("\nFinal test evaluation complete.")
    print(paths.evaluation_dir)


if __name__ == "__main__":
    main()
