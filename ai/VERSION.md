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

--------------------------------------------------

## v0.2.0 - Fine-Tuned Model Release

This version contains the first fine-tuned EfficientNetB0 model for the Rice Leaf Disease Detection System.

### Completed
- Improved the baseline EfficientNetB0 transfer learning model through fine-tuning.
- Reduced dropout and adjusted the classifier head.
- Fine-tuned selected EfficientNetB0 layers.
- Applied dynamic data augmentation during training.
- Used class weighting to improve weaker disease classes.
- Improved model performance on the internal test set.

### Internal Test Set Performance
- Accuracy: 90.00%
- Macro Precision: 90.82%
- Macro Recall: 90.00%
- Macro F1-score: 90.08%
- Macro ROC AUC: 0.9916
- Weighted ROC AUC: 0.9916

### Current Limitation
Although the model achieved strong internal test performance, testing with externally sourced internet images showed that the model still struggles with some real-world images. Some predictions were incorrect or had low to moderate confidence scores. Further fine-tuning and dataset improvement are needed before final validation.

### Next Target
The next version will focus on improving generalization using additional external images, reducing incorrect predictions, and improving confidence scores before conducting 5-fold cross-validation.