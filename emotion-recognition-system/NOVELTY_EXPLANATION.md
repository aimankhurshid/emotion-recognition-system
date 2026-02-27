# Research Project: Novelty & Performance Benchmarks (RAF-DB)

This document provides the specific "Novelty" and "Results" for your project to show your professor.

---

## 1. The Core Novelty: Temporal-Spatial Bi-LSTM
The base paper (Alzahrani et al., 2025) uses a Dual Attention Mechanism (DAM) to identify "What" and "Where" features are important in a face. However, it treats the face as a static grid.

**Our Improvement:** 
We treat the output of the Dual Attention module as a **sequence of spatial tokens**. By passing these tokens through a **Bi-directional Long Short-Term Memory (Bi-LSTM)**, the model learns the **coordination sequence** between different facial landmarks (e.g., how the eyes widen in synchronization with the mouth opening).

### Why this is scientifically superior:
- **Captures Micro-Expressions:** Subtle emotions like "Contempt" or "Disgust" involve asymmetric muscle movements that are hard to catch in a static frame but obvious when modeled as a sequence of features.
- **Contextual Awareness:** The Bi-directional nature allows the model to understand the relationship between the top of the face (eyebrows) and the bottom (jaw) simultaneously.

---

## 2. Table-1: Comparison on RAF-DB Dataset

| Method | Backbone | Dataset | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| SCN (Duan et al., 2024) | DarkNet-19 | RAF-DB | 87.03% | Benchmark |
| RAN (Li et al., 2023) | VGGNet | RAF-DB | 86.90% | Benchmark |
| EfficientFace (2024) | Custom | RAF-DB | 85.12% | Benchmark |
| **DCD-DAN (Base Paper)** | **ResNet50** | **RAF-DB** | **93.18%** | **SOTA** |
| **Our Proposed (DAM + Bi-LSTM)** | **ResNet50** | **RAF-DB** | **[target]** | **Target** |

---

## 3. The "Killer Argument" for the Review
When the professor asks: *"Why is your model better than the 2025 base paper?"*

**Your Answer:**
> "Ma'am, in the base paper's confusion matrix, they achieve 93% average accuracy, but they fail on **Contempt (reaching only 28%)**. This is because their model is static. Our Bi-LSTM novelty treats the spatial grid as a sequence, allowing the model to detect the **asymmetric muscle activation** required for Contempt. We expect to boost Contempt recognition and improve overall accuracy on RAF-DB."

---

## 4. Key Metrics Equations (To show in PPT)

1. **Accuracy:** $\frac{TP+TN}{TP+TN+FP+FN}$
2. **Precision (Exactness):** $\frac{TP}{TP+FP}$
3. **Recall (Completeness):** $\frac{TP}{TP+FN}$
4. **F1-Score (The Balance):** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 5. Technical Implementation (Model Code)
The novelty is implemented in `models/cnn_dual_attention_bilstm.py`.

```python
# The Novelty: Sequence Modeling of Spatial Features
self.lstm = nn.LSTM(
    input_size=feature_dim,
    hidden_size=lstm_hidden,
    num_layers=lstm_layers,
    bidirectional=True,  # Capture context from both sides
    batch_first=True
)
```
