# Final Project Roadmap: Beat the 82.13% Baseline

Follow these steps to complete the training on your RTX 5000 Ada laptop using the official AffectNet+ dataset.

---

## 🚀 Step 1: Dataset Setup (AffectNet+)
1.  **Extract:** Ensure your AffectNet+ images are extracted into the `data/` folder.
2.  **Organize:** Your directory must follow this structure:
    ```text
    data/
    ├── train/
    │   ├── 0_neutral/ (containing .jpg/.png files)
    │   ├── ...
    │   └── 7_contempt/
    ├── val/
    └── test/
    ```
    *(If you only have one folder, split it 70/15/15 into train/val/test).*

---

## 💻 Step 2: The Winning Training Command
To beat the 82.13% baseline, use this highly optimized command on the RTX 5000:

```bash
python training/train.py \
  --data_dir data \
  --epochs 60 \
  --batch_size 32 \
  --model_type full \
  --backbone efficientnet_b4 \
  --learning_rate 0.0001 \
  --lstm_hidden 512 \
  --lstm_layers 2 \
  --num_workers 8 \
  --use_class_weights \
  --checkpoint_dir results/winning_run_affectnet
```

### Why this is the "Winner":
*   **EfficientNet-B4**: Superior feature extraction compared to ResNet.
*   **LSTM Hidden 512**: Doubled hidden size to capture deeper temporal patterns in AffectNet+ facial features.
*   **Weighted Loss**: Handles the class imbalance (few 'Contempt' vs many 'Happy' images).
*   **Num Workers 8**: Takes advantage of the RTX 5000's CPU power for faster data loading.

---

## 📈 Step 3: Verification
Once training finishes:
1.  Run the evaluation: `python training/evaluate.py --checkpoint_path results/winning_run_affectnet/best_model.pth --data_dir data`
2.  Verify the **Validation Accuracy** is above **82.13%**.
3.  Check the `results/winning_run_affectnet/training_history.png` to ensure loss was steadily decreasing.

---

## 🎬 Step 4: Real-time Demo
Use your newly trained model for the webcam:
```bash
python webcam_demo_ultra.py --model_path results/winning_run_affectnet/best_model.pth
```
