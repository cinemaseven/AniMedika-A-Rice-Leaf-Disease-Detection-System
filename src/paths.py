from dataclasses import dataclass
from pathlib import Path

from config import EXPERIMENT_MODELS_DIR, EXPERIMENT_OUTPUTS_DIR


@dataclass(frozen=True)
class ExperimentPaths:
    name: str
    output_root: Path
    model_root: Path
    kfold_dir: Path
    training_dir: Path
    evaluation_dir: Path
    calibration_dir: Path
    best_model_path: Path
    final_model_path: Path
    labels_path: Path
    temperature_path: Path


def get_experiment_paths(name: str) -> ExperimentPaths:
    output_root = EXPERIMENT_OUTPUTS_DIR / name
    model_root = EXPERIMENT_MODELS_DIR / name
    paths = ExperimentPaths(
        name=name,
        output_root=output_root,
        model_root=model_root,
        kfold_dir=output_root / "kfold",
        training_dir=output_root / "training",
        evaluation_dir=output_root / "evaluation",
        calibration_dir=output_root / "calibration",
        best_model_path=model_root / "best_model.keras",
        final_model_path=model_root / "final_model.keras",
        labels_path=model_root / "labels.json",
        temperature_path=model_root / "temperature.json",
    )
    for folder in [
        paths.output_root,
        paths.model_root,
        paths.kfold_dir,
        paths.training_dir,
        paths.evaluation_dir,
        paths.calibration_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    return paths
