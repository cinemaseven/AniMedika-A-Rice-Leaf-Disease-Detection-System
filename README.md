# AniMedika

**AniMedika** is a browser-based rice leaf disease detection system developed as an undergraduate thesis project. It uses an EfficientNetB0-based Convolutional Neural Network (CNN) to classify a submitted rice leaf image into one of six supported conditions and presents disease information, key signs, predefined English and Filipino recommendations, and season-based notes.

The system is intended as a supplementary identification and decision-support tool and does not replace diagnosis or advice from qualified agricultural professionals.

## Features

- Upload an existing rice leaf image or capture one using a compatible mobile device
- Classify six rice leaf conditions
- English and Filipino interface and disease information
- Predefined next-step recommendations
- Automatic wet- or dry-season information based on the selected image date
- Lightweight rule-based NLP for extracting key disease-related phrases from disease descriptions
- Source links for additional disease information
- In-memory image processing without permanent backend storage
- Docker support for local and production deployment

## Supported Classes

The model performs single-label multiclass classification across the following classes:

- Bacterial Leaf Blight
- Brown Spot
- Healthy
- Rice Blast
- Sheath Blight
- Tungro

## Technology Stack

- **Model:** EfficientNetB0, TensorFlow/Keras
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Gunicorn
- **Supporting libraries:** NumPy, scikit-learn, Pillow, Pillow-HEIF

## Model and System Overview

The deployed classifier uses EfficientNetB0 with ImageNet-pretrained weights and a custom six-class classification head. Submitted images are corrected for orientation, converted to RGB, and resized to **224 × 224 pixels** before inference.

The deployed model corresponds to:

```text
models/experiments/exp04_finetune40/
```

The selected model uses temperature scaling from `temperature.json` for probability calibration during inference.

On the reserved 180-image test set, the selected model achieved:

- **Accuracy:** 82.78%
- **Macro F1-Score:** 82.70%
- **Macro ROC AUC:** 95.93%

### Prediction Flow

```text
Upload or capture image
        ↓
Validate and preprocess image
        ↓
EfficientNetB0 inference
        ↓
Apply temperature scaling
        ↓
Select predicted condition
        ↓
Retrieve disease information, recommendations, and seasonal note
        ↓
Extract key disease-related phrases
        ↓
Display results in the browser
```

Recommendations are predefined in `backend/recommendations.json`. The NLP component does not generate recommendations; it extracts key phrases from the disease descriptions presented by the system.

## Project Structure

```text
AniMedika/
├── backend/                  # Flask API, model inference, recommendations, NLP
├── css/                      # Website styles
├── images/                   # Website images and icons
├── js/                       # Frontend behavior and translations
├── models/
│   └── experiments/
│       └── exp04_finetune40/ # Selected deployed model
├── src/                      # Model training and evaluation scripts
├── scripts/                  # Utility scripts
├── compose.yaml
├── Dockerfile
├── home.html
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

- Python 3.13
- Git
- A modern web browser
- Docker Desktop, if using Docker

No Node.js installation is required for the current frontend.

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd AniMedika-A-Rice-Leaf-Disease-Detection-System
```

### 2. Install Python Dependencies

```powershell
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r requirements.txt
```

### 3. Confirm the Model Files

The following files should be present:

```text
models/experiments/exp04_finetune40/best_model.keras
models/experiments/exp04_finetune40/labels.json
models/experiments/exp04_finetune40/temperature.json
```

### 4. Run the Website Locally

From the repository root:

```powershell
py -3.13 -m backend.app
```

Open:

```text
http://localhost:8000
```

The Flask application serves both the frontend and backend API.

## Running with Docker

Build and start the application:

```powershell
docker compose up --build -d
```

Open:

```text
http://localhost:8080
```

Useful Docker commands:

```powershell
docker compose ps
docker compose logs -f
docker compose down
```

## API Endpoints

### `GET /api/health`

Returns the server and model status.

### `POST /api/predict`

Accepts a multipart image upload and selected image date. The response includes the predicted condition, seasonal context, localized disease information, extracted key phrases, predefined recommendations, and related source links.

## Image Handling and Privacy

Supported image formats include JPEG, PNG, WEBP, HEIC, and HEIF, with a maximum upload size of **10 MB**.

Submitted images are processed in memory for prediction and are not permanently saved by the backend. The prediction response identifies uploaded images as `stored: false`.

## Limitations

- The model supports only the six listed classes.
- The system performs single-label classification and returns only one predicted condition per image.
- It does not independently verify whether an uploaded image is a rice leaf.
- Images containing unsupported diseases or unrelated objects may still be assigned one of the available classes.
- Lighting, blur, obstruction, background complexity, image angle, and cropping may affect prediction quality.
- The model is intended as a supplementary tool and not as a substitute for professional agricultural diagnosis.

## Research Project

AniMedika was developed as part of the undergraduate thesis:

**Rice Leaf Disease Detection Using an EfficientNetB0-Based Convolutional Neural Network with Multi-Metric Performance Evaluation**

Detailed dataset preparation, controlled experiments, five-fold cross-validation, multi-seed evaluation, calibration analysis, class-level results, and other research evidence are documented separately in the thesis and supporting research materials rather than duplicated in this README.

## License and Attribution

Dataset sources, pretrained model resources, libraries, icons, images, and other third-party materials remain subject to their respective licenses and attribution requirements. A project-level software license should be added if the repository will be made available for reuse or redistribution.
