from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from config import CLASS_NAMES
from dataset import get_split_dataset
from experiments import get_experiment
from metrics import calculate_metrics, collect_predictions, save_json
from paths import get_experiment_paths


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    log_probabilities = np.log(np.clip(probabilities, 1e-8, 1.0))
    scaled = log_probabilities / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def fit_temperature(y_true_one_hot: np.ndarray, probabilities: np.ndarray) -> float:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    y_true_tensor = tf.constant(y_true_one_hot, dtype=tf.float32)
    logits_tensor = tf.constant(logits, dtype=tf.float32)
    log_temperature = tf.Variable(0.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    for _ in range(300):
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            calibrated = tf.nn.softmax(logits_tensor / temperature, axis=1)
            loss = loss_function(y_true_tensor, calibrated)
        gradients = tape.gradient(loss, [log_temperature])
        optimizer.apply_gradients(zip(gradients, [log_temperature]))
    return float(tf.exp(log_temperature).numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--source", choices=["oof", "validation"], default="oof")
    args = parser.parse_args()

    experiment = get_experiment(args.experiment)
    paths = get_experiment_paths(args.experiment)
    model = tf.keras.models.load_model(paths.best_model_path)

    if args.source == "oof":
        predictions_path = paths.kfold_dir / "oof_predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError("Run kfold.py first or use --source validation")
        frame = pd.read_csv(predictions_path)
        y_true = frame["actual_class"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()
        probabilities = frame[[f"probability_{name}" for name in CLASS_NAMES]].to_numpy()
    else:
        dataset, _ = get_split_dataset("validation", shuffle=False, seed=experiment["seed"])
        y_true, _, probabilities = collect_predictions(model, dataset)

    y_true_one_hot = np.eye(len(CLASS_NAMES))[y_true]
    temperature = fit_temperature(y_true_one_hot, probabilities)
    calibrated = apply_temperature(probabilities, temperature)

    raw_pred = np.argmax(probabilities, axis=1)
    calibrated_pred = np.argmax(calibrated, axis=1)
    raw_metrics, _ = calculate_metrics(y_true, raw_pred, probabilities)
    calibrated_metrics, _ = calculate_metrics(y_true, calibrated_pred, calibrated)

    paths.temperature_path.write_text(
        json.dumps({"temperature": temperature, "fit_source": args.source}, indent=4),
        encoding="utf-8",
    )
    comparison = pd.DataFrame(
        [
            {"probability_type": "raw", **raw_metrics},
            {"probability_type": "temperature_scaled", **calibrated_metrics},
        ]
    )
    comparison.to_csv(paths.calibration_dir / "calibration_comparison.csv", index=False)
    save_json(
        {
            "temperature": temperature,
            "fit_source": args.source,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
        },
        paths.calibration_dir / "calibration_results.json",
    )
    print(f"Temperature: {temperature:.6f}")
    print(f"Saved: {paths.temperature_path}")
    print(f"Comparison: {paths.calibration_dir / 'calibration_comparison.csv'}")


if __name__ == "__main__":
    main()
