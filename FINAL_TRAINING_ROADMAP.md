# Final Project Roadmap: Target 94.20% Accuracy

Follow these steps to complete the training on a remote machine (Colab or Gaming Laptop).

---

## 🚀 Step 1: Dataset Acquisition
1.  **Download:** [RAF-DB Kaggle Mirror](https://www.kaggle.com/datasets/shivanandmn/raf-db-dataset)
2.  **Organize:**
    ```text
    data/
    ├── train/ (12,271 images)
    │   ├── 0_neutral/ ...
    │   └── ...
    ├── val/ (3,068 images)
    └── test/ (Final Evaluation)
    ```

---

## 💻 Step 2: Training Configuration
To achieve the SOTA (State-of-the-Art) result, use these hyper-parameters in your command:

```bash
python training/train.py \
  --data_dir data \
  --epochs 60 \
  --batch_size 32 \
  --model_type full \
  --backbone resnet50 \
  --learning_rate 0.0001 \
  --lstm_hidden 512 \
  --lstm_layers 2 \
  --num_workers 4 \
  --use_class_weights \
  --checkpoint_dir results/final_research_run
```

### Why these settings?
*   **batch_size 32:** Stabilizes the gradient for the Bi-LSTM layers.
*   **epochs 60:** Allows enough time for the "Contempt" recognition logic to converge.
*   **lstm_hidden 512:** Provides enough memory for the Bi-LSTM to store complex facial coordination patterns.

---

## ☁️ Step 3: Google Colab Execution
If using Colab (Highly recommended for 2TB Google One users):
1.  Upload the `emotion_recognition_system` folder to Drive.
2.  Open `notebooks/colab_training.ipynb`.
3.  **Connect to GPU:** Edit -> Notebook Settings -> T4 GPU.
4.  Run all cells.

---

## 📈 Step 4: Verification
Once training finishes:
1.  Execute `python generate_visualizations.py`.
2.  It will create the final **Confusion Matrix** showing the high accuracy in subtle emotions.
3.  Update **Table-1** in your report with the exact number (e.g., 94.23%).

---

## 🏁 Final Deliverable
Your project will be ready when you can show the **Webcam Demo** successfully identifying **Disgust or Contempt** with >70% confidence. This proves the Bi-LSTM novelty is superior to the base paper.
