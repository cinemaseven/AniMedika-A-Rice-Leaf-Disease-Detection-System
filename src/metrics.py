from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from config import CLASS_NAMES, NUM_CLASSES


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_probs: list[np.ndarray] = []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_probs.extend(probabilities)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    true_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(y_probs, dtype=float)
    predicted_array = np.argmax(probability_array, axis=1)
    return true_array, predicted_array, probability_array


def expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    bins: int = 10,
) -> float:
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    correctness = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(correctness[mask])
        bin_confidence = np.mean(confidences[mask])
        error += np.mean(mask) * abs(bin_accuracy - bin_confidence)
    return float(error)


def multiclass_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    one_hot = np.eye(NUM_CLASSES)[y_true]
    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class_rows: list[dict] = []
    total = matrix.sum()
    specificities: list[float] = []
    for index, class_name in enumerate(CLASS_NAMES):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        specificities.append(float(specificity))
        per_class_rows.append(
            {
                "class": class_name,
                "precision": float(precision[index]),
                "recall_sensitivity": float(recall[index]),
                "specificity": float(specificity),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
            }
        )

    accuracy = accuracy_score(y_true, y_pred)
    confidences = np.max(y_probs, axis=1)
    results = {
        "sample_count": int(len(y_true)),
        "correct_count": int(np.sum(y_true == y_pred)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity_macro": float(np.mean(specificities)),
        "specificity_weighted": float(np.average(specificities, weights=support)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_probs, labels=labels)),
        "brier_score_multiclass": multiclass_brier_score(y_true, y_probs),
        "mean_confidence": float(np.mean(confidences)),
        "confidence_accuracy_gap": float(np.mean(confidences) - accuracy),
        "expected_calibration_error_10_bins": expected_calibration_error(y_true, y_probs, bins=10),
    }

    one_hot = label_binarize(y_true, classes=labels)
    try:
        results["roc_auc_ovr_macro"] = float(
            roc_auc_score(one_hot, y_probs, average="macro", multi_class="ovr")
        )
        results["roc_auc_ovr_weighted"] = float(
            roc_auc_score(one_hot, y_probs, average="weighted", multi_class="ovr")
        )
    except ValueError:
        results["roc_auc_ovr_macro"] = None
        results["roc_auc_ovr_weighted"] = None

    return results, pd.DataFrame(per_class_rows)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(ax=axis, xticks_rotation=45, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_roc_curve(y_true, y_probs, path: Path, title: str) -> None:
    one_hot = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    figure, axis = plt.subplots(figsize=(10, 8))
    for index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(one_hot[:, index], y_probs[:, index])
        score = auc(fpr, tpr)
        axis.plot(fpr, tpr, label=f"{class_name}: {score:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_classification_report(y_true, y_pred, path: Path) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    path.write_text(report, encoding="utf-8")


def save_predictions(
    y_true,
    y_pred,
    y_probs,
    path: Path,
    image_paths: list[str] | None = None,
    extra_columns: dict[str, list] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index in range(len(y_true)):
        row = {
            "image_path": image_paths[index] if image_paths and index < len(image_paths) else "",
            "actual_class": CLASS_NAMES[int(y_true[index])],
            "predicted_class": CLASS_NAMES[int(y_pred[index])],
            "correct": bool(y_true[index] == y_pred[index]),
            "confidence": float(np.max(y_probs[index])),
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            row[f"probability_{class_name}"] = float(y_probs[index, class_index])
        if extra_columns:
            for name, values in extra_columns.items():
                row[name] = values[index]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def save_metric_bundle(
    model,
    dataset,
    output_dir: Path,
    prefix: str,
    image_paths: list[str] | None = None,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_probs = collect_predictions(model, dataset)
    metrics, per_class = calculate_metrics(y_true, y_pred, y_probs)
    save_json(metrics, output_dir / f"{prefix}_metrics.json")
    per_class.to_csv(output_dir / f"{prefix}_per_class_metrics.csv", index=False)
    save_confusion_matrix(
        y_true,
        y_pred,
        output_dir / f"{prefix}_confusion_matrix.png",
        f"{prefix.replace('_', ' ').title()} Confusion Matrix",
    )
    save_classification_report(
        y_true,
        y_pred,
        output_dir / f"{prefix}_classification_report.txt",
    )
    save_roc_curve(
        y_true,
        y_probs,
        output_dir / f"{prefix}_roc_curve.png",
        f"{prefix.replace('_', ' ').title()} ROC Curves",
    )
    save_predictions(
        y_true,
        y_pred,
        y_probs,
        output_dir / f"{prefix}_predictions.csv",
        image_paths=image_paths,
    )
    return metrics, per_class, y_true, y_pred, y_probs


HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "specificity_macro",
    "specificity_weighted",
    "cohen_kappa",
    "matthews_correlation_coefficient",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
}
LOWER_IS_BETTER = {
    "log_loss",
    "brier_score_multiclass",
    "expected_calibration_error_10_bins",
    "confidence_accuracy_gap",
}


def compare_metric_sets(
    first_name: str,
    first: dict,
    second_name: str,
    second: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    common = sorted((HIGHER_IS_BETTER | LOWER_IS_BETTER) & first.keys() & second.keys())
    for metric in common:
        first_value = first.get(metric)
        second_value = second.get(metric)
        if first_value is None or second_value is None:
            continue
        if metric in HIGHER_IS_BETTER:
            generalization_gap = float(first_value - second_value)
            meaning = "positive means the first split scored higher"
        else:
            generalization_gap = float(second_value - first_value)
            meaning = "positive means the second split has higher error"
        rows.append(
            {
                "comparison": f"{first_name}_vs_{second_name}",
                "metric": metric,
                f"{first_name}_value": float(first_value),
                f"{second_name}_value": float(second_value),
                "generalization_gap": generalization_gap,
                "absolute_difference": abs(float(first_value - second_value)),
                "interpretation": meaning,
            }
        )
    return pd.DataFrame(rows)
