from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Final selected model
DEFAULT_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "exp04_finetune40"
)

DEFAULT_MODEL_PATH = (
    DEFAULT_EXPERIMENT_DIRECTORY
    / "best_model.keras"
)

DEFAULT_CLASS_NAMES_PATH = (
    DEFAULT_EXPERIMENT_DIRECTORY
    / "labels.json"
)

DEFAULT_TEMPERATURE_PATH = (
    DEFAULT_EXPERIMENT_DIRECTORY
    / "temperature.json"
)

MODEL_PATH = Path(
    os.getenv(
        "ANIMEDIKA_MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)

CLASS_NAMES_PATH = Path(
    os.getenv(
        "ANIMEDIKA_CLASS_NAMES_PATH",
        str(DEFAULT_CLASS_NAMES_PATH),
    )
)

TEMPERATURE_PATH = Path(
    os.getenv(
        "ANIMEDIKA_TEMPERATURE_PATH",
        str(DEFAULT_TEMPERATURE_PATH),
    )
)

RECOMMENDATIONS_PATH = Path(
    os.getenv(
        "ANIMEDIKA_RECOMMENDATIONS_PATH",
        str(
            PROJECT_ROOT
            / "backend"
            / "recommendations.json"
        ),
    )
)

RECOMMENDATION_SOURCES_PATH = Path(
    os.getenv(
        "ANIMEDIKA_RECOMMENDATION_SOURCES_PATH",
        str(
            PROJECT_ROOT
            / "backend"
            / "recommendation_sources.json"
        ),
    )
)

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "HEIF",
    "HEIC",
}

INPUT_SCALE_MODE = os.getenv(
    "ANIMEDIKA_INPUT_SCALE_MODE",
    "efficientnet_internal",
)