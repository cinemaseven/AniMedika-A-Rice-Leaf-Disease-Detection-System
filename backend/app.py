from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from backend.config import MAX_UPLOAD_BYTES, PROJECT_ROOT
from backend.image_processing import InvalidImageError, prepare_image_for_model
from backend.model_service import (
    ModelNotReadyError,
    PredictionError,
    is_model_ready,
    predict_preprocessed_image,
    # warm_up_model,
)
from backend.recommendation_service import (
    RecommendationError,
    determine_season,
    get_result_bundle,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

@app.get("/")
def home():
    return send_from_directory(PROJECT_ROOT, "home.html")

@app.get("/css/<path:filename>")
def css_asset(filename: str):
    return send_from_directory(PROJECT_ROOT / "css", filename)

@app.get("/js/<path:filename>")
def js_asset(filename: str):
    return send_from_directory(PROJECT_ROOT / "js", filename)

@app.get("/images/<path:filename>")
def image_asset(filename: str):
    return send_from_directory(PROJECT_ROOT / "images", filename)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_ready": is_model_ready(),
            "storage": "in_memory_only",
        }
    )

@app.post("/api/predict")
def predict():
    uploaded_file = request.files.get("image")
    selected_date = request.form.get("selected_date", "")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please provide an image."}), 400

    try:
        season = determine_season(selected_date)
        image_bytes = uploaded_file.read()
        prepared = prepare_image_for_model(image_bytes)
        prediction = predict_preprocessed_image(prepared.batch)
        localized_result = get_result_bundle(prediction["disease_id"], season)

        return jsonify(
            {
                "prediction": prediction,
                "context": {
                    "selected_date": selected_date,
                    "season": season,
                },
                "image": {
                    "original_width": prepared.original_width,
                    "original_height": prepared.original_height,
                    "format": prepared.image_format,
                    "stored": False,
                },
                "localized_result": localized_result,
            }
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except InvalidImageError as exc:
        return jsonify({"error": str(exc)}), 415
    except ModelNotReadyError as exc:
        return (
            jsonify(
                {
                    "error": "The final model is not ready yet.",
                    "detail": str(exc),
                    "code": "MODEL_NOT_READY",
                }
            ),
            503,
        )
    except (PredictionError, RecommendationError) as exc:
        app.logger.exception("Prediction pipeline error")
        return jsonify({"error": str(exc)}), 500

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The image is too large. Maximum size is 10 MB."}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)