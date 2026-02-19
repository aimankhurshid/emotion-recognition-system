# Lenovo RTX Laptop: Winning Training Run Guide

Follow these exact steps to start the baseline-beating training on your friend's laptop.

## 1. Environment Setup (Windows/NVIDIA)
Open **PowerShell** as Administrator and run:

```powershell
# 1. Clone the professional version of your repo
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion-recognition-system

# 2. Create and Activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# 3. Install PyTorch with CUDA 12.1 (Best for RTX 4000 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install -r requirements.txt
```

## 2. Dataset Preparation (AffectNet+)
1.  Copy your **5 ZIP parts** to the laptop.
2.  Use **7-Zip** to extract `AffectNet+_part1.zip` (it should automatically pull from the other 4 parts).
3.  Ensure the folders are in `data/train`, `data/val`, etc.
    *   *Verify: You should have subfolders like `0_neutral`, `7_contempt` inside `data/train`.*

## 3. The Winning Training Command
Run this command to start the ~12-24 hour training. This uses the **EfficientNet-B4 + 512 Hidden LSTM** strategy:

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
  --num_workers 4 `
  --use_class_weights `
  --checkpoint_dir results/winning_run_rtx_laptop
```

## 4. Tips for your Friend's Laptop
- **Plug in the Charger**: Laptops throttle GPU power by 50-80% when unplugged.
- **Keep it Cool**: Lift the back of the laptop or use a cooling pad. This is a heavy training run.
- **Disable Sleep**: Go to Power Settings and set "Sleep" to **Never** while plugged in.

---
*For full technical details, see `docs/antigravity/implementation_plan.md`.*
