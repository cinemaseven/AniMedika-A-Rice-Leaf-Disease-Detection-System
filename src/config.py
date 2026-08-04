from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SRC_DIR.parent

DATASET_DIR = PYTHON_DIR / "dataset"
RAW_DATASET_DIR = DATASET_DIR
SPLIT_DATASET_DIR = DATASET_DIR / "split"
TRAIN_DIR = SPLIT_DATASET_DIR / "train"
VALIDATION_DIR = SPLIT_DATASET_DIR / "validation"
TEST_DIR = SPLIT_DATASET_DIR / "test"
SPLIT_MANIFEST_PATH = SPLIT_DATASET_DIR / "split_manifest.csv"
SPLIT_SUMMARY_PATH = SPLIT_DATASET_DIR / "split_summary.csv"

MODELS_DIR = PYTHON_DIR / "models"
OUTPUTS_DIR = PYTHON_DIR / "outputs"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments"
EXPERIMENT_OUTPUTS_DIR = OUTPUTS_DIR / "experiments"

AUDIT_DIR = OUTPUTS_DIR / "dataset_audit"
IMAGE_INVENTORY_PATH = AUDIT_DIR / "image_inventory.csv"
EXACT_DUPLICATES_PATH = AUDIT_DIR / "exact_duplicates.csv"
NEAR_DUPLICATE_REVIEW_PATH = AUDIT_DIR / "near_duplicate_review.csv"
DATASET_GROUPS_PATH = AUDIT_DIR / "dataset_groups.csv"
AUDIT_SUMMARY_PATH = AUDIT_DIR / "audit_summary.json"

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

# Potential near-duplicate candidates are produced when their dHash
# Hamming distance is less than or equal to this value.
NEAR_DUPLICATE_THRESHOLD = 4

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
        AUDIT_DIR,
        EXPERIMENT_MODELS_DIR,
        EXPERIMENT_OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_required_folders()
    print("Required folders created.")