from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import applications, layers, models, optimizers

from augmentation import get_data_augmentation
from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES


def _make_optimizer(training_settings: dict, learning_rate: float):
    name = training_settings.get("optimizer", "adam").lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=training_settings.get("weight_decay", 1e-5),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def compile_model(model: tf.keras.Model, experiment: dict, learning_rate: float) -> None:
    training = experiment["training"]
    model.compile(
        optimizer=_make_optimizer(training, learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=training.get("label_smoothing", 0.0)
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def build_model(experiment: dict, trainable_base: bool = False) -> tf.keras.Model:
    model_settings = experiment["model"]
    inputs = layers.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        name="input_image",
    )
    x = get_data_augmentation(experiment["augmentation"])(inputs)

    backbone = applications.EfficientNetB0(
        include_top=False,
        weights=model_settings.get("weights", "imagenet"),
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    )
    backbone.trainable = trainable_base

    # Keep EfficientNet BatchNormalization behavior stable during transfer learning.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(model_settings["dropout_1"], name="dropout_1")(x)
    x = layers.Dense(
        model_settings["dense_units"],
        activation="relu",
        name=f"dense_{model_settings['dense_units']}",
    )(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(model_settings["dropout_2"], name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="AniMedika_EfficientNetB0")
    compile_model(
        model,
        experiment,
        experiment["training"]["initial_learning_rate"],
    )
    return model


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("EfficientNetB0 backbone was not found")


def enable_fine_tuning(model: tf.keras.Model, experiment: dict) -> tf.keras.Model:
    backbone = get_backbone(model)
    unfreeze_last = int(experiment["model"]["unfreeze_last_layers"])
    backbone.trainable = True

    for layer in backbone.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in backbone.layers[-unfreeze_last:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    compile_model(
        model,
        experiment,
        experiment["training"]["fine_tune_learning_rate"],
    )
    return model
