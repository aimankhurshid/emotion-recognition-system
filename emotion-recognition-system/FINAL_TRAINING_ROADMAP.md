# Final Project Roadmap: The 82.13% Baseline Challenge

This project follows a two-phase research strategy to validate the Bi-LSTM novelty before the final high-precision run.

---

## 🏎️ Phase 1: Benchmark Run (Lenovo RTX 4060 Laptop)
**Goal**: Verify model convergence and get preliminary accuracy.
**Constraint**: 8GB VRAM (Requires lower batch size).

```powershell
python training/train.py `
  --data_dir data `
  --epochs 30 `
  --batch_size 16 `
  --model_type full `
  --backbone efficientnet_b4 `
  --lstm_hidden 512 `
  --use_class_weights `
  --checkpoint_dir results/phase1_laptop_benchmark
```

---

## 🚀 Phase 2: Winning Run (University RTX 5000 Ada PC)
**Goal**: Beat the 82.13% baseline with maximum precision.
**Benefit**: 32GB VRAM allows for larger batch processing and longer training.

```powershell
python training/train.py `
  --data_dir data `
  --epochs 60 `
  --batch_size 32 `
  --model_type full `
  --backbone efficientnet_b4 `
  --learning_rate 0.0001 `
  --lstm_hidden 512 `
  --lstm_layers 2 `
  --num_workers 8 `
  --use_class_weights `
  --checkpoint_dir results/phase2_winning_run
```

---

## 📈 Verification Steps (Applies to both)
1.  **Evaluate**: `python training/evaluate.py --checkpoint_path <path_to_best_model> --data_dir data`
2.  **Webcam**: `python webcam_demo_ultra.py --model_path <path_to_best_model>`
3.  **Baseline Check**: Compare your Phase 2 Val Accuracy against **82.13%**.
