import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnet_model(num_classes, image_size=(224, 224)):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
            layers.RandomBrightness(0.05),
        ],
        name="data_augmentation"
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))

    x = data_augmentation(inputs)

    # EfficientNet in Keras already includes preprocessing.
    # Do not divide image values by 255.
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model