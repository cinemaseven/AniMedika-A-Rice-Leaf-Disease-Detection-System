from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image

AI_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = AI_DIR / "saved_model" / "rice_model.keras"
CLASS_NAMES_PATH = AI_DIR / "saved_model" / "class_names.json"

IMAGE_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    return predicted_class, confidence


if __name__ == "__main__":
    image_path = input("Enter image path: ")

    predicted_class, confidence = predict_image(image_path)

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")