from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from metrics import calculate_metrics, collect_predictions
from model import build_model, enable_fine_tuning


def get_callbacks(experiment: dict, checkpoint_path: Path):
    settings = experiment["training"]
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=settings["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=settings["reduce_lr_factor"],
            patience=settings["reduce_lr_patience"],
            min_lr=settings["minimum_learning_rate"],
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def history_to_frame(history, phase: str, starting_epoch: int) -> pd.DataFrame:
    frame = pd.DataFrame(history.history)
    frame.insert(0, "phase_epoch", range(1, len(frame) + 1))
    frame.insert(0, "global_epoch", range(starting_epoch, starting_epoch + len(frame)))
    frame.insert(2, "phase", phase)
    if {"accuracy", "val_accuracy"}.issubset(frame.columns):
        frame["accuracy_gap_train_minus_validation"] = frame["accuracy"] - frame["val_accuracy"]
    if {"loss", "val_loss"}.issubset(frame.columns):
        frame["loss_gap_validation_minus_train"] = frame["val_loss"] - frame["loss"]
    return frame


def save_history_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = int((frame["phase"] == "frozen_backbone").sum())

    for metric, validation_metric, ylabel, filename in [
        ("accuracy", "val_accuracy", "Accuracy", "training_validation_accuracy.png"),
        ("loss", "val_loss", "Categorical Cross-Entropy", "training_validation_loss.png"),
    ]:
        if metric not in frame or validation_metric not in frame:
            continue
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(frame["global_epoch"], frame[metric], label=f"Training {ylabel}")
        axis.plot(frame["global_epoch"], frame[validation_metric], label=f"Validation {ylabel}")
        if boundary > 0 and boundary < len(frame):
            axis.axvline(boundary + 0.5, linestyle="--", label="Fine-tuning begins")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Training and Validation {ylabel}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=300)
        plt.close(figure)

    gap_columns = [
        column
        for column in [
            "accuracy_gap_train_minus_validation",
            "loss_gap_validation_minus_train",
        ]
        if column in frame
    ]
    if gap_columns:
        figure, axis = plt.subplots(figsize=(9, 6))
        for column in gap_columns:
            axis.plot(frame["global_epoch"], frame[column], label=column.replace("_", " ").title())
        axis.axhline(0, linestyle="--", label="No difference")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Gap")
        axis.set_title("Training–Validation Gaps by Epoch")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "training_validation_epoch_gaps.png", dpi=300)
        plt.close(figure)


def _validation_metrics(model, validation_dataset) -> dict:
    y_true, y_pred, y_probs = collect_predictions(model, validation_dataset)
    metrics, _ = calculate_metrics(y_true, y_pred, y_probs)
    return metrics


def train_two_phase(
    experiment: dict,
    train_dataset,
    validation_dataset,
    output_dir: Path,
) -> tuple[tf.keras.Model, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint = output_dir / "phase1_best.keras"
    phase2_checkpoint = output_dir / "phase2_best.keras"

    model = build_model(experiment, trainable_base=False)
    history1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=experiment["training"]["initial_epochs"],
        callbacks=get_callbacks(experiment, phase1_checkpoint),
        verbose=1,
    )

    phase1_model = tf.keras.models.load_model(phase1_checkpoint)
    phase1_metrics = _validation_metrics(phase1_model, validation_dataset)

    model = enable_fine_tuning(phase1_model, experiment)
    history2 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=experiment["training"]["fine_tune_epochs"],
        callbacks=get_callbacks(experiment, phase2_checkpoint),
        verbose=1,
    )
    model.save(output_dir / "phase2_final.keras")

    phase2_model = tf.keras.models.load_model(phase2_checkpoint)
    phase2_metrics = _validation_metrics(phase2_model, validation_dataset)

    candidates = [
        ("phase1", phase1_model, phase1_metrics, phase1_checkpoint),
        ("phase2", phase2_model, phase2_metrics, phase2_checkpoint),
    ]
    candidates.sort(
        key=lambda item: (
            item[2]["f1_macro"],
            item[2]["accuracy"],
            -item[2]["log_loss"],
        ),
        reverse=True,
    )
    selected_phase, selected_model, selected_metrics, selected_path = candidates[0]
    best_model_path = output_dir / "best_model.keras"
    shutil.copy2(selected_path, best_model_path)

    # Reload the selected checkpoint from disk.
    # This prevents an in-memory model modified during fine-tuning
    # from being returned when Phase 1 was selected.
    selected_model = tf.keras.models.load_model(best_model_path)

    selection_frame = pd.DataFrame(
        [
            {
                "phase": phase,
                "selected": phase == selected_phase,
                **metrics,
            }
            for phase, _, metrics, _ in [
                ("phase1", phase1_model, phase1_metrics, phase1_checkpoint),
                ("phase2", phase2_model, phase2_metrics, phase2_checkpoint),
            ]
        ]
    )
    selection_frame.to_csv(output_dir / "phase_selection.csv", index=False)

    frame1 = history_to_frame(history1, "frozen_backbone", 1)
    frame2 = history_to_frame(history2, "fine_tuning", len(frame1) + 1)
    history_frame = pd.concat([frame1, frame2], ignore_index=True)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    save_history_plots(history_frame, output_dir)

    selection = {
        "selected_phase": selected_phase,
        "selected_validation_metrics": selected_metrics,
        "phase1_validation_metrics": phase1_metrics,
        "phase2_validation_metrics": phase2_metrics,
        "best_model_path": str(best_model_path),
    }
    return selected_model, history_frame, selection
