python training/evaluate.py \
  --checkpoint_path /Users/ayeman/Downloads/rafdb_epoch21_acc0.897_bacc0.8275.pth \
  --data_dir data \
  --output_dir results/visualizations# Paper Outline - RAF-DB Pivot

## 1. Introduction
- Problem: robust facial emotion recognition in the wild.
- Gap: base paper focuses on spatial attention only; temporal or sequential feature modeling is limited.
- Goal: improve recognition on RAF-DB with a small, well-justified modification.

## 2. Literature Review (4-6 papers)
- Base paper: Alzahrani et al., 2025 (DCD-DAN).
- Include recent FER papers with dataset + accuracy notes.

## 3. Proposed Method
- Backbone: EfficientNet-B4 or ResNet50 (state which one you use).
- Dual attention: channel + spatial attention.
- Novelty: Bi-LSTM over spatial tokens to model feature coordination.
- Provide a block diagram and short description of data flow.

## 4. Dataset Description
- Dataset: RAF-DB (single dataset for direct comparison).
- Justification (2-3 lines): RAF-DB is a large, real-world, in-the-wild FER benchmark with standard emotion classes and wide usage in recent literature, making it suitable for fair comparison and generalization testing.
- Provide class list and split counts used in your experiments.

## 5. Experimental Results
- Metrics: Accuracy, Precision, Recall, F1-score.
- Table-1: same structure as base paper, but only RAF-DB column retained.

Table 1. Comparison on RAF-DB
| Method | Backbone | Dataset | Accuracy | Notes |
| --- | --- | --- | --- | --- |
| SCN (2024) | DarkNet-19 | RAF-DB | [value] | baseline |
| RAN (2023) | VGGNet | RAF-DB | [value] | baseline |
| DCD-DAN (Base) | ResNet50 | RAF-DB | 93.18% | base paper |
| Our Method | [Backbone] + Bi-LSTM | RAF-DB | [value] | proposed |

- Add confusion matrix and short error analysis.

## 6. Conclusion
- Summarize the novelty and its effect on RAF-DB.
- State that results follow the base paper table format for fair comparison.

## 7. References
- Base paper + 4-6 recent FER works.
