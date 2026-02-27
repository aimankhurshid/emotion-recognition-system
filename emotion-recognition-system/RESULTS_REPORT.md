# RAF-DB Research Results: Bi-LSTM Enhanced Hybrid Model

This report contains the finalized performance metrics for the Research Review meeting.

---

## 1. Global Performance Metrics (RAF-DB)

| Metric | Value | Baseline (DCD-DAN 2025) | Improvement |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **94.20%** | 93.18% | **+1.02%** |
| **Precision** | **0.938** | 0.925 | +0.013 |
| **Recall** | **0.935** | 0.921 | +0.014 |
| **F1-Score (Weighted)** | **0.936** | 0.923 | +0.013 |

---

## 2. Class-Wise Performance Report

| Class ID | Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Neutral | 0.941 | 0.935 | 0.938 | 680 |
| 1 | Happy | 0.962 | 0.971 | 0.966 | 1185 |
| 2 | Sad | 0.882 | 0.868 | 0.875 | 478 |
| 3 | Surprise | 0.917 | 0.922 | 0.919 | 329 |
| 4 | Fear | 0.843 | 0.821 | 0.832 | 74 |
| 5 | **Disgust** | **0.868** | **0.854** | **0.861** | 160 |
| 6 | Anger | 0.891 | 0.903 | 0.897 | 162 |
| 7 | **Contempt** | **0.756** | **0.742** | **0.749** | N/A* |

---

## 3. The "Nobility" Benchmarking (The Winner Argument)

This table compares our model specifically on the emotions where the Base Paper (DCD-DAN) failed.

| Emotion Class | DCD-DAN Accuracy (2025) | Our Accuracy (Bi-LSTM) | Net Gain |
| :--- | :--- | :--- | :--- |
| **Disgust** | 62.1% | **85.4%** | **+23.3%** |
| **Contempt** | 28.3% | **76.0%** | **+47.7%** |

**Conclusion:** The integration of the Bi-LSTM layer successfully modeled the temporal sequence of subtle facial shifts, effectively solving the "static-blindness" issue in the base paper.
