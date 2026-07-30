# AniMedika: A Browser-Based Rice Leaf Disease Detection System

AniMedika is a browser-based rice leaf disease detection system developed to identify six rice leaf conditions using an EfficientNetB0-based Convolutional Neural Network (CNN). The system allows users to upload or capture a rice leaf image, receive a predicted class and confidence score, and view corresponding disease-management recommendations.

The project focuses on rice leaf diseases that affect crop productivity and food security, particularly in rice-producing regions of Luzon, Philippines.

> **Important:** AniMedika is a decision-support tool. Its prediction should not replace field inspection, laboratory testing, or consultation with a qualified agricultural professional.

---

## Project Overview

Rice leaf diseases such as Rice Blast, Bacterial Leaf Blight, Brown Spot, Sheath Blight, and Tungro can reduce rice yield and negatively affect food security. Manual disease identification may be difficult when symptoms appear visually similar or when trained agricultural personnel are not immediately available.

AniMedika applies deep learning and transfer learning to assist with rice leaf disease identification through a standard web browser. The system uses an EfficientNetB0 CNN trained on labeled rice leaf images. It produces one predicted class, a user-facing confidence score, and disease-management information.

The final model supports the following six classes:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy
4. Rice Blast
5. Sheath Blight
6. Tungro

The platform is intended for farmers, agricultural practitioners, students, researchers, and other users who need an accessible rice leaf disease screening tool without installing specialized desktop software.

---

## Research Objectives

The project aims to:

- Develop a browser-based rice leaf disease detection system.
- Apply an EfficientNetB0-based CNN for multiclass image classification.
- Use transfer learning to improve performance with a limited agricultural image dataset.
- Apply image augmentation to improve generalization.
- Evaluate the model using multiple metrics rather than accuracy alone.
- Compare controlled model experiments using five-fold cross-validation.
- Provide users with a predicted condition, confidence score, and management recommendations.
- Support deployment through standard web browsers and Docker-based environments.

---

## Supported Classes

|          Class          | Description |
|-------------------------|-------------|
| `Bacterial_Leaf_Blight` | A bacterial rice disease commonly associated with leaf wilting, yellowing, and blighted areas. |
| `Brown_Spot`            | A fungal disease that may appear as brown, circular, or oval lesions on the leaf. |
| `Healthy`               | Rice leaves without visible symptoms belonging to the five supported disease classes. |
| `Rice_Blast`            | A fungal disease that can produce spindle-shaped lesions and may affect different parts of the rice plant. |
| `Sheath_Blight`         | A fungal disease that commonly affects the leaf sheath and may spread upward under favorable conditions. |
| `Tungro`                | A viral rice disease associated with yellow-orange discoloration, stunting, and reduced vigor. |

The model performs **single-label multiclass classification**. Each uploaded image is assigned to one of the six supported classes.

---

## System Features

- Browser-based user interface
- Image upload support
- Camera capture support, depending on browser permissions and device compatibility
- Six-class rice leaf condition classification
- EfficientNetB0 transfer-learning model
- Real-time inference through a Python backend
- User-facing confidence score
- Confidence display capped at 99.9%
- English and Filipino disease-management recommendations
- Docker support for consistent local and hosted deployment
- Separate research and deployment files
- Multi-metric model evaluation
- Experiment logging and reproducibility support

---

## Model Architecture

The selected model uses EfficientNetB0 as its convolutional feature extractor.

### Final Architecture

```text
Input image: 224 × 224 × 3 RGB
        ↓
Data augmentation
        ↓
EfficientNetB0
- ImageNet pretrained weights
- include_top=False
        ↓
Global Average Pooling
        ↓
Batch Normalization
        ↓
Dropout: 0.40
        ↓
Dense: 256 units, ReLU
        ↓
Batch Normalization
        ↓
Dropout: 0.30
        ↓
Dense: 6 units, Softmax
```

### Key Configuration

|           Component          | Setting |
|------------------------------|---------|
| Backbone                     | EfficientNetB0 |
| Pretrained weights           | ImageNet |
| Input size                   | 224 × 224 × 3 |
| Classification type          | Single-label multiclass |
| Hidden dense layer           | 256 units |
| Hidden activation            | ReLU |
| Output layer                 | 6 units |
| Output activation            | Softmax |
| Loss function                | Categorical cross-entropy |
| Optimizer                    | Adam |
| Phase 1 learning rate        | `1e-4` |
| Phase 2 learning rate tested | `1e-5` |
| Batch size                   | 16 |
| Phase 1 maximum epochs       | 25 |
| Phase 2 maximum epochs       | 15 |
| Selected final phase         | Phase 1, frozen backbone |
| Selected experiment          | `exp08_vertical_translation003` |

EfficientNetB0 was selected because it offers a practical balance between classification performance and computational efficiency for server-side web deployment.

---

## Dataset and Data Splitting

The complete dataset contains 900 labeled rice leaf images, with 150 images per class.

| Partition | Images per class | Total | Percentage |
|---|---:|---:|---:|
| Training   |    105  |   630   |    70%   |
| Validation |    15   |    90   |    10%   |
| Testing    |    30   |   180   |    20%   |
| **Total**  | **150** | **900** | **100%** |

### Purpose of Each Partition

- **Training set:** Used for model fitting, controlled experiments, and five-fold cross-validation.
- **Validation set:** Used for final model monitoring, checkpoint selection, phase selection, and calibration experiments.
- **Test set:** Kept separate until the final selected model was evaluated.

The final 180-image test set was not used to train the model.

### Five-Fold Cross-Validation

Five-fold cross-validation was performed only on the 630-image training partition.

In each fold:

- Approximately 504 images were used for fold training.
- Approximately 126 images were used for fold validation.
- Every training-partition image appeared once in an out-of-fold validation subset.

The study did not select only the highest-performing fold. Model comparison used mean performance, standard deviation, out-of-fold predictions, per-class metrics, and confusion patterns across all five folds.

---

## Image Preprocessing and Augmentation

### Image Preprocessing

Each image is:

1. Opened using Pillow.
2. Converted to RGB.
3. Resized to 224 × 224 pixels.
4. Converted to a `float32` NumPy array.
5. Expanded into a batch with shape `(1, 224, 224, 3)` during single-image prediction.

### Selected Augmentation Configuration

| Augmentation | Setting |
|---|---|
| Horizontal flip         | Enabled |
| Vertical flip           | Disabled |
| Rotation                | `0.08` |
| Zoom                    | `0.10` |
| Contrast                | `0.10` |
| Vertical translation    | `0.03` |
| Horizontal translation  | `0.08` |
| Brightness augmentation | Disabled |

The selected augmentation was intended to improve tolerance to realistic variations in camera orientation, scale, contrast, and leaf position while reducing transformations that could excessively alter the natural vertical structure of rice leaves.

---

## Training Strategy

The model-development process used transfer learning in two possible phases.

### Phase 1: Frozen Backbone

- EfficientNetB0 weights were initialized from ImageNet.
- The EfficientNetB0 backbone remained frozen.
- Only the classification head was trained.
- Adam was used with a learning rate of `1e-4`.

### Phase 2: Fine-Tuning

- The final 30 EfficientNetB0 layers were considered for fine-tuning.
- Batch normalization layers remained frozen.
- Adam was used with a lower learning rate of `1e-5`.

### Final Phase Selection

For the final training run, Phase 1 achieved better validation accuracy and macro F1-score than Phase 2.

| Validation Metric | Phase 1 | Phase 2 |
|---|---:|---:|
| Accuracy    | **87.78%** | 86.67% |
| Macro F1    | **87.79%** | 86.64% |
| Log loss    |   0.2991   | **0.2538** |
| Brier score |   0.1623   | **0.1410** |

The predefined model-selection priority emphasized:

1. Macro F1-score
2. Accuracy
3. Log loss as a tie-breaker

Therefore, the Phase 1 checkpoint was selected as the final model.

---

## Experiment Selection

Nine controlled experiments were evaluated against the original EfficientNetB0 baseline.

| Experiment | Main change |
|---|---|
|   Baseline   | Horizontal and vertical flipping |
| Experiment 1 | Horizontal flip only |
| Experiment 2 | Milder geometry |
| Experiment 3 | Label smoothing of 0.05 |
| Experiment 4 | Fine-tune final 40 layers |
| Experiment 5 | Reduce dense layer to 128 units |
| Experiment 6 | Replace Adam with AdamW |
| Experiment 7 | Reduce rotation to 0.04 only |
| Experiment 8 | Reduce vertical translation to 0.03 only |
| Experiment 9 | Reduce fine-tuning learning rate to `5e-6` |

### Selected Experiment

The final experiment is:

```text
exp08_vertical_translation003
```

Experiment 8 retained horizontal-only flipping and reduced only the vertical translation factor from 0.08 to 0.03.

### Multi-Seed Comparison

Experiment 8 and Experiment 1 were compared using the same three fixed seeds:

```text
42, 123, 2026
```

| Metric | Experiment 1 | Experiment 8 |
|---|---:|---:|
| Mean accuracy          | 94.29% | **94.50%** |
| Accuracy SD            | 0.79% | **0.40%** |
| Mean macro F1          | 94.27% | **94.47%** |
| Macro F1 SD            | 0.81% | **0.42%** |
| Mean log loss          | 0.1743 | **0.1714** |
| Mean accuracy gap      | 3.58% | **3.44%** |
| Mean Rice Blast recall | **88.57%** | 88.25% |
| Mean Sheath Blight F1  | 91.86% | **92.49%** |

Experiment 8 was selected because it had higher average accuracy and macro F1, lower log loss, a smaller average generalization gap, and lower variation across seeds.

---

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-averaged precision, recall, and F1-score
- Confusion matrix
- ROC AUC
- Log loss
- Multiclass Brier score
- Expected Calibration Error
- Specificity
- Cohen's kappa
- Matthews Correlation Coefficient
- Training-validation-testing metric gaps

### Metric Definitions

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The confusion matrix was used to identify the exact direction of classification errors, particularly between Rice Blast and Sheath Blight.

---

## Final Model Performance

The selected model was evaluated once on the 180-image test set.

### Overall Results

| Metric | Result |
|---|---:|
| Correct predictions              | 170 of 180 |
| Incorrect predictions                | 10 |
| Accuracy                         | **94.44%** |
| Macro precision                  | **94.65%** |
| Macro recall                     | **94.44%** |
| Macro F1-score                   | **94.45%** |
| Specificity                      | **98.89%** |
| Cohen's kappa                    | **0.9333** |
| Matthews correlation coefficient | **0.9337** |
| Macro ROC AUC                    | **99.65%** |
| Log loss                         | **0.1640** |
| Multiclass Brier score           | **0.0870** |
| Expected Calibration Error       | **2.46%**  |

### Per-Class Results

| Class | Precision | Recall | F1-score | Correct |
|---|---:|---:|---:|---:|
| Bacterial Leaf Blight | 100.00% | 100.00% | 100.00% | 30/30 |
| Brown Spot            | 93.33% | 93.33% | 93.33% | 28/30 |
| Healthy               | 100.00% | 93.33% | 96.55% | 28/30 |
| Rice Blast            | 89.29% | 83.33% | 86.21% | 25/30 |
| Sheath Blight         | 85.29% | 96.67% | 90.63% | 29/30 |
| Tungro                | 100.00% | 100.00% | 100.00% | 30/30 |

### Final Error Distribution

```text
Brown Spot → Rice Blast:      2
Healthy → Brown Spot:         2
Rice Blast → Sheath Blight:   5
Sheath Blight → Rice Blast:   1
```

Six of the ten final errors involved the Rice Blast–Sheath Blight pair.

---

## Repository Structure

```text
AniMedika-A-Rice-Leaf-Disease-Detection-System/
│
├── backend/
├── frontend/
├── src/
│   ├── calibration.py
│   ├── compare_splits.py
│   ├── config.py
│   ├── evaluate.py
│   ├── kfold.py
│   ├── model.py
│   ├── paths.py
│   ├── predict.py
│   ├── train.py
│   └── training_utils.py
│
├── models/
│   └── experiments/
│       └── exp08_vertical_translation003/
│           ├── best_model.keras
│           └── labels.json
│
├── data/
│   └── recommendations.json
│
├── dataset/                 # Local only; ignored by Git
├── outputs/                 # Generated locally; ignored by Git
├── .venv/                   # Local environment; ignored by Git
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Generated datasets, outputs, prediction logs, alternate checkpoints, and temporary files should remain excluded from Git.

---

## Installation

### Prerequisites

- Python 3.13
- Git
- Node.js, when required by the frontend
- Docker Desktop, for container-based execution
- A modern browser

### Clone the Repository

```powershell
git clone <repository-url>
cd AniMedika-A-Rice-Leaf-Disease-Detection-System
```

### Create and Activate a Virtual Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install Python Dependencies

```powershell
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r requirements.txt
```

### Install Frontend Dependencies

```powershell
npm install
```

Run the Node.js step only when the repository contains a Node-based frontend.

---

## Running Predictions from the Command Line

From the repository root:

```powershell
py -3.13 src\predict.py
```

The script loads the model once and repeatedly requests an image path.

Example path:

```text
C:\Users\PC\Desktop\field_images\leaf01.jpg
```

Example output:

```text
Prediction: Rice_Blast
Confidence: 94.7%
Calibration applied: False
Temperature used: 1.000000
```

Enter `y` to predict another image or `n` to close the program.

Prediction records are generated under:

```text
outputs/experiments/exp08_vertical_translation003/predictions/
```

---

## Running the Website Locally

The exact command depends on the final backend and frontend entry points.

### Backend

```powershell
py -3.13 -m backend.app
```

or:

```powershell
py -3.13 backend\app.py
```

### Frontend

```powershell
npm run dev
```

For a production frontend build:

```powershell
npm run build
```

Use the command that matches the final repository structure.

---

## Running with Docker

### Build and Start

```powershell
docker compose up --build -d
```

### View Status

```powershell
docker compose ps
```

### View Logs

```powershell
docker compose logs -f
```

### Stop

```powershell
docker compose down
```

### Rebuild

```powershell
docker compose down
docker compose up --build -d
```

A rebuild is normally required after changing dependencies, the Dockerfile, or model files.

---

## Deployment Model Files

Deploy:

```text
models/experiments/exp08_vertical_translation003/best_model.keras
models/experiments/exp08_vertical_translation003/labels.json
```

Do not deploy:

- Raw or split datasets
- Cross-validation fold checkpoints
- Training histories
- Evaluation outputs
- Confusion matrices
- Old model files
- Prediction logs
- Temperature calibration files

The `.keras` file stores the model architecture and learned parameters. It does not contain the original training images.

---

## Prediction Flow

```text
User uploads or captures an image
        ↓
Browser sends the image to the backend
        ↓
Backend checks file type and size
        ↓
Image is converted to RGB
        ↓
Image is resized to 224 × 224
        ↓
EfficientNetB0 performs inference
        ↓
Softmax returns six probabilities
        ↓
The highest probability determines the class
        ↓
The backend retrieves recommendations
        ↓
The website displays the result
```

---

## Confidence Score Handling

The deployed system uses raw softmax confidence.

Temperature scaling was evaluated, but the fitted validation temperature worsened final test log loss, Brier score, and expected calibration error. It is therefore not applied in deployment.

Current policy:

- Preserve the genuine softmax probability internally.
- Cap only the displayed confidence at 99.9%.
- Do not interpret confidence as a guarantee of correctness.

Example:

```text
Actual probability:   0.999999
Displayed confidence: 99.9%
```

---

## Data and Privacy

The complete training dataset is not committed to GitHub. The repository contains the source code and selected model required to run the system.

The deployed backend should avoid permanently storing uploaded images unless users explicitly consent.

Recommended behavior:

- Process the image only for prediction.
- Do not store the original local file path.
- Do not log image contents.
- Avoid collecting precise location data.
- Delete temporary files after processing.
- Clearly disclose any future image-storage feature.

A database is not required for a stateless upload-predict-display workflow.

---

## Limitations

1. The model performs closed-set classification and always chooses one of the six supported classes.
2. It does not currently verify that the uploaded image is a rice leaf.
3. Conditions outside the six supported classes cannot be identified correctly.
4. Rice Blast and Sheath Blight remain the most frequently confused classes.
5. Poor lighting, blur, distance, obstruction, and extreme cropping may reduce reliability.
6. Performance may change on field images that differ from the development dataset.
7. High softmax confidence does not guarantee correctness.
8. Disease-management recommendations should be confirmed using local agricultural guidance.
9. The system is not a replacement for expert or laboratory diagnosis.

---

## Recommended Future Improvements

- Collect additional field images from different locations and devices.
- Increase the number of difficult Rice Blast and Sheath Blight examples.
- Perform external validation using independently collected images.
- Add a separate rice-leaf input validation stage.
- Add Grad-CAM or another explainability method.
- Add expert feedback and correction features.
- Test mobile camera behavior across Android and iOS.
- Conduct usability testing with farmers and agricultural practitioners.
- Evaluate latency and memory usage on the selected hosting platform.
- Expand localized and seasonal disease-management recommendations.

Future model changes should use a new development or external validation set rather than repeatedly tuning against the archived final test set.

---

## Troubleshooting

### `python` is not recognized

Use:

```powershell
py -3.13
```

### TensorFlow does not load

```powershell
py -3.13 -m pip show tensorflow
py -3.13 -m pip install -r requirements.txt
```

### Model file not found

Confirm:

```text
models/experiments/exp08_vertical_translation003/best_model.keras
```

### Labels file not found

Confirm:

```text
models/experiments/exp08_vertical_translation003/labels.json
```

### Image path contains spaces

Paste it in quotation marks:

```text
"C:\Users\PC\Desktop\field images\leaf01.jpg"
```

### Docker container exits

```powershell
docker compose logs -f
```

Common causes include a missing model, missing dependency, wrong port, invalid environment variable, or failed frontend build.

### Changes do not appear in Docker

```powershell
docker compose down
docker compose up --build -d
```

---

## Research Keywords

`rice leaf disease detection`  
`EfficientNet`  
`EfficientNetB0`  
`Convolutional Neural Network`  
`deep learning`  
`transfer learning`  
`browser-based system`  
`rice disease classification`  
`Luzon`  
`Philippines`

---

## Citation

A formal citation should be added when the thesis is finalized.

Temporary format:

```text
AniMedika Research Team. (2026). AniMedika: A browser-based rice leaf
disease detection system using EfficientNetB0 [Computer software].
```

---

## License

No license is declared in this README. Add a `LICENSE` file before allowing public reuse, redistribution, or modification. Verify the licenses of all datasets, pretrained weights, libraries, icons, images, and third-party resources.

---

## Project Status

- Model development: Completed
- Selected experiment: `exp08_vertical_translation003`
- Selected checkpoint: `best_model.keras`
- Final test evaluation: Completed
- Browser integration: In progress
- Docker integration: In progress
- Field validation and usability evaluation: In progress

---

## Acknowledgment

This project was developed as an undergraduate research system intended to support rice disease monitoring and informed crop-management decisions. The researchers acknowledge the contributions of agricultural experts, farmers, image-data contributors, open-source software communities, and academic advisers.
