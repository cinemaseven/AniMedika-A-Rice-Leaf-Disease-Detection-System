from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from config import SEED

BASELINE: dict[str, Any] = {
    "name": "baseline",
        "description": (
            "Initial reference EfficientNetB0 configuration used as the "
            "experimental control for evaluating succeeding model modifications "
            "under the group-aware 70/10/20 dataset protocol."
        ),
    "seed": SEED,
    "model": {
        "backbone": "EfficientNetB0",
        "weights": "imagenet",
        "dense_units": 256,
        "dropout_1": 0.40,
        "dropout_2": 0.30,
        "unfreeze_last_layers": 30,
    },
    "augmentation": {
        "flip_mode": "horizontal_and_vertical",
        "rotation": 0.08,
        "zoom": 0.10,
        "contrast": 0.10,
        "translation_height": 0.08,
        "translation_width": 0.08,
        "brightness": 0.00,
    },
    "training": {
        "optimizer": "adam",
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "initial_learning_rate": 1e-4,
        "fine_tune_learning_rate": 1e-5,
        "initial_epochs": 25,
        "fine_tune_epochs": 15,
        "early_stopping_patience": 7,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.30,
        "minimum_learning_rate": 1e-7,
    },
}


def _variant(name: str, description: str, updates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    experiment = copy.deepcopy(BASELINE)
    experiment["name"] = name
    experiment["description"] = description
    for section, values in updates.items():
        experiment[section].update(values)
    return experiment


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": BASELINE,
    "exp01_horizontal_only": _variant(
        "exp01_horizontal_only",
        "Tests removal of vertical flipping while keeping every other baseline setting fixed.",
        {"augmentation": {"flip_mode": "horizontal"}},
    ),
    "exp02_milder_geometry": _variant(
        "exp02_milder_geometry",
        "Tests orientation-preserving and milder geometric augmentation.",
        {
            "augmentation": {
                "flip_mode": "horizontal",
                "rotation": 0.04,
                "translation_height": 0.03,
                "translation_width": 0.05,
            }
        },
    ),
    "exp03_label_smoothing": _variant(
        "exp03_label_smoothing",
        (
            "Tests label smoothing 0.05 using the current "
            "horizontal-only winning augmentation configuration."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
            },
            "training": {
                "label_smoothing": 0.05,
            },
        },
    ),

    "exp04_finetune40": _variant(
        "exp04_finetune40",
        (
            "Tests fine-tuning the final 40 EfficientNetB0 "
            "layers instead of 30 using horizontal-only flipping."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
            },
            "model": {
                "unfreeze_last_layers": 40,
            },
        },
    ),

    "exp05_dense128": _variant(
        "exp05_dense128",
        (
            "Tests a smaller 128-unit classification head "
            "using the horizontal-only augmentation configuration."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
            },
            "model": {
                "dense_units": 128,
            },
        },
    ),

    "exp06_adamw": _variant(
        "exp06_adamw",
        (
            "Tests AdamW with weight decay 1e-5 using "
            "the horizontal-only augmentation configuration."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
            },
            "training": {
                "optimizer": "adamw",
                "weight_decay": 1e-5,
            },
        },
    ),

    "exp07_rotation004_only": _variant(
        "exp07_rotation004_only",
        (
            "Tests milder rotation while retaining the winning "
            "horizontal-only augmentation and original translation settings."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
                "rotation": 0.04,
            },
        },
    ),

    "exp08_vertical_translation003": _variant(
        "exp08_vertical_translation003",
        (
            "Tests reduced vertical translation while retaining the "
            "winning rotation and horizontal translation settings."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
                "translation_height": 0.03,
            },
        },
    ),

    "exp09_finetune_lr5e6": _variant(
        "exp09_finetune_lr5e6",
        (
            "Tests a lower fine-tuning learning rate while retaining "
            "the winning horizontal-only configuration."
        ),
        {
            "augmentation": {
                "flip_mode": "horizontal",
            },
            "training": {
                "fine_tune_learning_rate": 5e-6,
            },
        },
    ),
}


def get_experiment(name: str) -> dict[str, Any]:
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return copy.deepcopy(EXPERIMENTS[name])


def save_experiment_config(experiment: dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(experiment, file, indent=4)
