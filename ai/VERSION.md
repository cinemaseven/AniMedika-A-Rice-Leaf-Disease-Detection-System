# AI Model Version Log

## v0.1.0 - Baseline Model Release

This version contains the first working baseline model pipeline for the rice leaf disease detection system.

### Completed
- Organized rice leaf disease dataset by class.
- Resized images to 224x224.
- Split dataset into training and testing folders.
- Implemented baseline EfficientNetB0 transfer learning model.
- Trained initial baseline model.
- Saved model class names.
- Tested single-image prediction through predict.py.
- Generated initial confidence score outputs.

### Current Limitation
- The baseline model may still produce low-confidence predictions.
- Fine-tuning has not yet been applied.
- 5-fold cross-validation has not yet been completed.
- Backend and frontend integration are not yet included in this version.

### Next Target
v0.2.0 will focus on model improvement through fine-tuning, improved augmentation, full evaluation, and better prediction confidence.