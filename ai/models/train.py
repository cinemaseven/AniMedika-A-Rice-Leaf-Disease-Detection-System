from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from efficientnet import build_efficientnet_model

AI_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = AI_DIR / "dataset" / "split" / "train"
SAVED_MODEL_DIR = AI_DIR / "saved_model"
RESULTS_DIR = AI_DIR / "results"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
VALIDATION_SPLIT = 0.20
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

with open(SAVED_MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = build_efficientnet_model(num_classes=num_classes, image_size=IMAGE_SIZE)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVED_MODEL_DIR / "rice_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.savefig(RESULTS_DIR / "training_history.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig(RESULTS_DIR / "training_loss.png")
plt.close()

print("\nTraining completed.")
print(f"Best model saved to: {SAVED_MODEL_DIR / 'rice_model.keras'}")
print(f"Class names saved to: {SAVED_MODEL_DIR / 'class_names.json'}")