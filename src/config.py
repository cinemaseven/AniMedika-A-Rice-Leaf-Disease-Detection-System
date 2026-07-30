from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SRC_DIR.parent

DATASET_DIR = PYTHON_DIR / "dataset"
RAW_DATASET_DIR = DATASET_DIR
SPLIT_DATASET_DIR = DATASET_DIR / "split"
TRAIN_DIR = SPLIT_DATASET_DIR / "train"
VALIDATION_DIR = SPLIT_DATASET_DIR / "validation"
TEST_DIR = SPLIT_DATASET_DIR / "test"

MODELS_DIR = PYTHON_DIR / "models"
OUTPUTS_DIR = PYTHON_DIR / "outputs"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments"
EXPERIMENT_OUTPUTS_DIR = OUTPUTS_DIR / "experiments"

CLASS_NAMES = [
    "Bacterial_Leaf_Blight",
    "Brown_Spot",
    "Healthy",
    "Rice_Blast",
    "Sheath_Blight",
    "Tungro",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
BATCH_SIZE = 16
SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20
N_SPLITS = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_required_folders() -> None:
    for path in [
        DATASET_DIR,
        SPLIT_DATASET_DIR,
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        EXPERIMENT_MODELS_DIR,
        EXPERIMENT_OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_required_folders()
    print("Required folders created.")
