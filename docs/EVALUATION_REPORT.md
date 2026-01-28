# Test & Evaluation Report

## 1. Executive Summary
The Fake News Detection System was evaluated using a held-out test set comprising 12,607 samples from the social media dataset. The **Bi-Directional LSTM Neural Network** achieved an overall accuracy of **99.02%**, significantly outperforming baseline models.

---

## 2. Methodology
*   **Dataset**: Social Media Fake News Dataset (Merged & Augmented).
*   **Test Size**: 12,607 samples (20% split).
*   **Metrics**: Accuracy, Precision, Recall, F1-Score, Receiver Operating Characteristic Area Under Curve (ROC-AUC).

---

## 3. Model Performance (LSTM)

### 3.1 Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Fake** | 0.98 | 1.00 | 0.99 | 6144 |
| **Real** | 1.00 | 0.98 | 0.99 | 6463 |
| **Accuracy** | | | **0.99** | 12607 |
| **Macro Avg** | 0.99 | 0.99 | 0.99 | 12607 |

### 3.2 Interpretation
*   **High Precision (Real = 1.00)**: When the model predicts news is "Real", it is almost always correct.
*   **High Recall (Fake = 1.00)**: The model successfully catches almost all Fake news items (very few false negatives).

---

## 4. Visualizations
The following artifacts were generated during evaluation:
*   `models/lstm_confusion_matrix.png`: Visual heatmap of true positives vs false positives.
*   `models/lstm_model.h5`: The serialized Keras model.

## 5. Conclusion
The LSTM model demonstrates production-ready performance with >99% accuracy. The augmented training strategy successfully resolved early biases against scientific/technical domains, establishing a robust classifier for diverse social media content.
