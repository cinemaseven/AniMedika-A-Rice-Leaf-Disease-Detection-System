from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np

from backend.config import (
    CLASS_NAMES_PATH, 
    MODEL_PATH, 
    TEMPERATURE_PATH,
)

class ModelNotReadyError(RuntimeError):
    """Raised when the trained model or its labels are missing."""

class PredictionError(RuntimeError):
    """Raised when the model cannot be loaded or cannot produce a valid prediction."""

_model = None
_class_names: list[str] | None = None
_temperature = 1.0
_calibration_applied = False
_load_lock = Lock()
_prediction_lock = Lock()

MAX_DISPLAY_CONFIDENCE = 0.999


def _load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise ModelNotReadyError(f"Class names file is missing: {path}")

    with path.open("r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ModelNotReadyError("The class names file is invalid.")

    return names

def _load_temperature(path: Path,) -> tuple[float, bool]:
    if not path.exists():
        return 1.0, False

    try:
        with path.open("r", encoding="utf-8",) as file:
            data = json.load(file)

        temperature = float(data["temperature"])

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise PredictionError("The temperature calibration file is invalid.") from exc

    if temperature <= 0:
        raise PredictionError("Temperature must be greater than zero.")

    return temperature, True


def _apply_temperature(probabilities: np.ndarray, temperature: float,) -> np.ndarray:
    log_probabilities = np.log(
        np.clip(
            probabilities,
            1e-8,
            1.0,
        )
    )

    scaled = log_probabilities / temperature

    scaled -= np.max(
        scaled,
        axis=1,
        keepdims=True,
    )

    exponentiated = np.exp(scaled)

    return exponentiated / np.sum(
        exponentiated,
        axis=1,
        keepdims=True,
    )

def is_model_ready() -> bool:
    return MODEL_PATH.is_file() and CLASS_NAMES_PATH.is_file()


def load_model_if_needed():
    global _model
    global _class_names
    global _temperature
    global _calibration_applied

    if _model is not None and _class_names is not None:
        return _model, _class_names, _temperature, _calibration_applied

    if not MODEL_PATH.exists():
        raise ModelNotReadyError(f"The trained model is missing: {MODEL_PATH}")

    with _load_lock:
        if _model is None:
            try:
                import tensorflow as tf

                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                class_names = _load_class_names(CLASS_NAMES_PATH)
                temperature, calibration_applied = _load_temperature(TEMPERATURE_PATH)

                input_shape = tuple(model.input_shape)
                output_units = int(model.output_shape[-1])

                if input_shape != (None, 224, 224, 3):
                    raise PredictionError(
                        "Unexpected model input shape: "
                        f"{input_shape}; expected (None, 224, 224, 3)."
                    )

                if output_units != len(class_names):
                    raise PredictionError(
                        "The model output count does not match labels.json: "
                        f"{output_units} outputs versus {len(class_names)} labels."
                    )

                _model = model
                _class_names = class_names
                _temperature = temperature
                _calibration_applied = (calibration_applied)
            except (ModelNotReadyError, PredictionError):
                raise
            except Exception as exc:
                raise PredictionError(f"The trained model could not be loaded: {exc}") from exc

    return _model, _class_names, _temperature, _calibration_applied

# ADDED FOR MODEL LOADING
# def warm_up_model() -> None:
#     """Load the model and perform one harmless inference."""

#     model, _class_names = load_model_if_needed()

#     try:
#         dummy_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)

#         # training=False prevents layers from changing their learned state.
#         model(dummy_batch, training=False)

#     except Exception as exc:
#         raise PredictionError(f"The model could not be warmed up: {exc}") from exc
# ---
    
def predict_preprocessed_image(batch: np.ndarray) -> dict:
    model, class_names, temperature, calibration_applied = load_model_if_needed()

    if batch.shape != (1, 224, 224, 3):
        raise PredictionError(
            f"Expected image batch shape (1, 224, 224, 3), received {batch.shape}."
        )

    try:
        with _prediction_lock:
            probabilities = np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float64)
    except Exception as exc:
        raise PredictionError(f"The model could not predict this image: {exc}") from exc

    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise PredictionError("The model output does not match the saved class-name order.")

    if not np.all(np.isfinite(probabilities)):
        raise PredictionError("The model returned invalid values.")
    
    if calibration_applied:
        probabilities = _apply_temperature(probabilities[None, :],temperature,)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.clip(probabilities[predicted_index], 0.0, 1.0))

    return {
        "class_index": predicted_index,
        "disease_id": class_names[predicted_index],
        "confidence": confidence,
        "display_confidence": min(confidence, MAX_DISPLAY_CONFIDENCE),
        "temperature_used": temperature,
        "calibration_applied": calibration_applied,
    }
