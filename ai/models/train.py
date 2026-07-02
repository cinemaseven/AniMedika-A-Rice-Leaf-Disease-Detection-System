from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from efficientnet import build_efficientnet_model

AI_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = AI_DIR / "dataset" / "split" / "train"
SAVED_MODEL_DIR = AI_DIR / "saved_model"
RESULTS_DIR = AI_DIR / "results"

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.20

INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 25

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

print("Class names:", class_names)

CLASS_WEIGHTS_BY_NAME = {
    "Bacterial_Leaf_Blight": 1.25,
    "Rice_Blast": 1.15,
    "Brown_Spot": 1.00,
    "Healthy": 1.00,
    "Sheath_Blight": 0.95,
    "Tungro": 1.00
}

class_weight = {
    index: CLASS_WEIGHTS_BY_NAME.get(class_name, 1.0)
    for index, class_name in enumerate(class_names)
}

print("Class weights:", class_weight)

with open(SAVED_MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model, base_model = build_efficientnet_model(
    num_classes=num_classes,
    image_size=IMAGE_SIZE
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVED_MODEL_DIR / "rice_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print("\n==============================")
print("STAGE 1: Training classifier head")
print("==============================\n")

history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight
)

print("\n==============================")
print("STAGE 2: Fine-tuning EfficientNetB0")
print("==============================\n")

base_model.trainable = True

fine_tune_at = len(base_model.layers) - 80

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(history_1.history["loss"]),
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight
)

accuracy = history_1.history.get("accuracy", []) + history_2.history.get("accuracy", [])
val_accuracy = history_1.history.get("val_accuracy", []) + history_2.history.get("val_accuracy", [])
loss = history_1.history.get("loss", []) + history_2.history.get("loss", [])
val_loss = history_1.history.get("val_loss", []) + history_2.history.get("val_loss", [])

history_df = pd.DataFrame({
    "epoch": list(range(1, len(accuracy) + 1)),
    "accuracy": accuracy,
    "val_accuracy": val_accuracy,
    "loss": loss,
    "val_loss": val_loss
})

history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

plt.figure()
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Fine-Tuned Model: Training and Validation Accuracy")
plt.savefig(RESULTS_DIR / "training_history.png")
plt.close()

plt.figure()
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Fine-Tuned Model: Training and Validation Loss")
plt.savefig(RESULTS_DIR / "training_loss.png")
plt.close()

print("\nTraining and fine-tuning completed.")
print(f"Best model saved to: {SAVED_MODEL_DIR / 'rice_model.keras'}")
print(f"Class names saved to: {SAVED_MODEL_DIR / 'class_names.json'}")
print(f"Training history saved to: {RESULTS_DIR / 'training_history.csv'}")