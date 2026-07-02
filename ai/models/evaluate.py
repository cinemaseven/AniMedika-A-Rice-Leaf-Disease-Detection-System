from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

AI_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = AI_DIR / "dataset" / "split" / "test"
MODEL_PATH = AI_DIR / "saved_model" / "rice_model.keras"
CLASS_NAMES_PATH = AI_DIR / "saved_model" / "class_names.json"
RESULTS_DIR = AI_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred_probs = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_pred_probs.extend(predictions)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

macro_roc_auc = roc_auc_score(
    y_true_bin,
    y_pred_probs,
    average="macro",
    multi_class="ovr"
)

weighted_roc_auc = roc_auc_score(
    y_true_bin,
    y_pred_probs,
    average="weighted",
    multi_class="ovr"
)

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curve.png")
plt.close()

metrics = {
    "macro_roc_auc": [macro_roc_auc],
    "weighted_roc_auc": [weighted_roc_auc]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

print(f"Macro ROC AUC: {macro_roc_auc:.4f}")
print(f"Weighted ROC AUC: {weighted_roc_auc:.4f}")
print("\nEvaluation completed.")