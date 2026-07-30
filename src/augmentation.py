import tensorflow as tf
from tensorflow.keras import layers


def get_data_augmentation(settings: dict) -> tf.keras.Sequential:
    augmentation_layers: list[tf.keras.layers.Layer] = []

    flip_mode = settings.get("flip_mode")
    if flip_mode and flip_mode != "none":
        augmentation_layers.append(layers.RandomFlip(flip_mode))

    if settings.get("rotation", 0) > 0:
        augmentation_layers.append(layers.RandomRotation(settings["rotation"]))
    if settings.get("zoom", 0) > 0:
        augmentation_layers.append(layers.RandomZoom(settings["zoom"]))
    if settings.get("contrast", 0) > 0:
        augmentation_layers.append(layers.RandomContrast(settings["contrast"]))
    if settings.get("brightness", 0) > 0:
        augmentation_layers.append(
            layers.RandomBrightness(settings["brightness"], value_range=(0, 255))
        )
    if settings.get("translation_height", 0) > 0 or settings.get("translation_width", 0) > 0:
        augmentation_layers.append(
            layers.RandomTranslation(
                height_factor=settings.get("translation_height", 0),
                width_factor=settings.get("translation_width", 0),
            )
        )

    return tf.keras.Sequential(augmentation_layers, name="data_augmentation")
