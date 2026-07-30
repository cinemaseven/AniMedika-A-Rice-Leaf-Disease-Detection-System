from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from calibration import apply_temperature
from config import CLASS_NAMES, IMAGE_SIZE
from paths import get_experiment_paths


# Default model configuration used for prediction.
DEFAULT_EXPERIMENT = "exp08_vertical_translation003"

# Maximum confidence displayed to the user.
# The genuine probability remains unchanged in the saved JSON file.
MAX_DISPLAY_CONFIDENCE = 0.999


def load_image(path: Path) -> np.ndarray:
    """Load and prepare one image for model prediction."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

    return np.expand_dims(
        image_array,
        axis=0,
    )


def get_display_confidence(confidence: float) -> float:
    """
    Return a user-facing confidence value capped at 99.9%.

    The actual model probability is not changed.
    """

    confidence = float(confidence)
    confidence = max(
        0.0,
        min(confidence, 1.0),
    )

    return min(
        confidence,
        MAX_DISPLAY_CONFIDENCE,
    )


def clean_image_path(user_input: str) -> Path:
    """
    Clean a path pasted from Windows Explorer or PowerShell.

    This removes surrounding single or double quotation marks.
    """

    cleaned = user_input.strip()

    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1]

    return Path(cleaned).expanduser()


def load_temperature(paths) -> tuple[float, bool]:
    """
    Load temperature scaling when temperature.json exists.

    Since you renamed temperature.json, the current model will use
    raw softmax probabilities and return temperature 1.0.
    """

    if not paths.temperature_path.exists():
        return 1.0, False

    temperature_data = json.loads(
        paths.temperature_path.read_text(
            encoding="utf-8",
        )
    )

    temperature = float(
        temperature_data["temperature"]
    )

    return temperature, True


def predict_image(
    model: tf.keras.Model,
    image_path: Path,
    experiment_name: str,
    paths,
    temperature: float,
    calibration_applied: bool,
) -> tuple[dict, Path]:
    """Predict one image and save its complete result."""

    probabilities = model.predict(
        load_image(image_path),
        verbose=0,
    )[0]

    # Apply temperature scaling only when temperature.json exists.
    if calibration_applied:
        probabilities = apply_temperature(
            probabilities[None, :],
            temperature,
        )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Genuine raw or calibrated model probability.
    raw_confidence = float(
        probabilities[predicted_index]
    )

    # User-facing value capped at 99.9%.
    display_confidence = get_display_confidence(
        raw_confidence
    )

    result = {
        "experiment": experiment_name,
        "image_path": str(
            image_path.resolve()
        ),
        "prediction": predicted_class,

        # Genuine confidence retained for records.
        "confidence": raw_confidence,
        "confidence_percent": round(
            raw_confidence * 100,
            6,
        ),

        # User-facing confidence capped below 100%.
        "display_confidence": display_confidence,
        "display_confidence_percent": round(
            display_confidence * 100,
            1,
        ),

        "temperature_used": temperature,
        "calibration_applied": calibration_applied,

        # Genuine probabilities for all six classes.
        "all_probabilities": {
            class_name: float(
                probabilities[class_index]
            )
            for class_index, class_name
            in enumerate(CLASS_NAMES)
        },
    }

    prediction_dir = (
        paths.output_root
        / "predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Microseconds prevent files from being overwritten when
    # multiple predictions occur within the same second.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    save_path = (
        prediction_dir
        / f"prediction_{timestamp}.json"
    )

    save_path.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    return result, save_path


def ask_to_continue() -> bool:
    """Ask whether the user wants to predict another image."""

    while True:
        answer = input(
            "\nPredict another image? (y/n): "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no", "q", "quit", "exit"}:
            return False

        print(
            "Please enter y for yes or n for no."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously predict rice-leaf images "
            "without restarting the program."
        )
    )

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=(
            "Experiment model to use. "
            f"Default: {DEFAULT_EXPERIMENT}"
        ),
    )

    args = parser.parse_args()

    paths = get_experiment_paths(
        args.experiment
    )

    if not paths.best_model_path.exists():
        raise FileNotFoundError(
            "The trained model was not found at:\n"
            f"{paths.best_model_path}"
        )

    print("\nLoading model. Please wait...")

    # The model is loaded only once.
    model = tf.keras.models.load_model(
        paths.best_model_path
    )

    temperature, calibration_applied = (
        load_temperature(paths)
    )

    print("\nModel loaded successfully.")
    print(f"Experiment: {args.experiment}")
    print(
        "Confidence type: "
        + (
            "temperature-scaled"
            if calibration_applied
            else "raw softmax"
        )
    )
    print(
        "Type q at the image-path prompt to exit."
    )

    while True:
        try:
            user_input = input(
                "\nPaste the full image path: "
            ).strip()

            if user_input.lower() in {
                "q",
                "quit",
                "exit",
            }:
                break

            if not user_input:
                print(
                    "No image path was entered."
                )
                continue

            image_path = clean_image_path(
                user_input
            )

            if not image_path.exists():
                print(
                    "\nImage file was not found:"
                )
                print(image_path)
                continue

            if not image_path.is_file():
                print(
                    "\nThe supplied path is not a file:"
                )
                print(image_path)
                continue

            try:
                result, save_path = predict_image(
                    model=model,
                    image_path=image_path,
                    experiment_name=args.experiment,
                    paths=paths,
                    temperature=temperature,
                    calibration_applied=(
                        calibration_applied
                    ),
                )

            except UnidentifiedImageError:
                print(
                    "\nThe selected file is not a "
                    "supported image."
                )
                continue

            except OSError as error:
                print(
                    "\nThe image could not be opened:"
                )
                print(error)
                continue

            print("\n------------------------------")
            print(
                f"Prediction: "
                f"{result['prediction']}"
            )
            print(
                f"Confidence: "
                f"{result['display_confidence_percent']:.1f}%"
            )
            print(
                f"Calibration applied: "
                f"{result['calibration_applied']}"
            )
            print(
                f"Temperature used: "
                f"{result['temperature_used']:.6f}"
            )
            print(f"Saved: {save_path}")
            print("------------------------------")

            if not ask_to_continue():
                break

        except KeyboardInterrupt:
            print()
            break

    print("\nPrediction program closed.")


if __name__ == "__main__":
    main()