# Training on University PC: Setup Guide

University PCs often have powerful GPUs (NVIDIA RTX series) which are much faster than laptops. Follow these steps to set up and train your project there.

---

## 🏗️ Step 1: Transfer the Project
1.  **USB Drive**: Copy your entire `emotion_recognition_system` folder to a USB drive.
2.  **Git**: Alternatively, if the Uni PC has internet, you can just clone your GitHub repo:
    ```bash
    git clone https://github.com/aimankhurshid/emotion-recognition-system.git
    ```

---

## 🐍 Step 2: Environment Setup (One-Time)
Most University PCs have Python installed. Open the **Command Prompt** (on Windows) or **Terminal** (on Linux/Mac) and run:

1.  **Create a Virtual Environment** (to avoid permission issues):
    ```bash
    python -m venv uni_env
    # Activate it:
    # On Windows: 
    uni_env\Scripts\activate
    # On Linux/Mac: 
    source uni_env/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Check for GPU (Crucial)**:
    University PCs usually have NVIDIA GPUs. If so, reinstall PyTorch with CUDA support for 10x speed:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

---

## 📂 Step 3: Dataset Preparation
1.  Copy your `data` folder (containing RAF-DB) into the project directory on the Uni PC.
2.  If you don't have it on USB, download it from Kaggle directly on the Uni PC to save time.

---

## 🚀 Step 4: Run Training
The command is the same as the roadmap. Use a larger batch size (e.g., 64) if the Uni PC has a powerful GPU (like an RTX 3060 or 4070).

```bash
python training/train.py \
  --data_dir data \
  --epochs 60 \
  --batch_size 64 \
  --model_type full \
  --backbone resnet50 \
  --learning_rate 0.0001 \
  --use_class_weights \
  --checkpoint_dir results/uni_pc_run
```

---

## ⚠️ Important Tips for University PCs
1.  **Don't save on Desktop**: University PCs often "wipe" the Desktop after logout. Save your work on the **D: drive** or a **network drive** (Z: drive) if provided.
2.  **Stay Logged In**: Training 60 epochs might take 2-4 hours. Check if the PC has an "Auto-Logoff" policy. Moving the mouse occasionally or using an "Auto-Clicker" can keep it awake.
3.  **Copy Checkpoints Back**: When finished, **immediately copy** the `results/uni_pc_run/` folder back to your USB drive or push it to GitHub. This folder contains the `.pth` file you need for your demo!

---

### 💡 Why this is great for your Review:
Tell your professor: *"Ma'am, I am training the final high-resolution weights on the University's workstation GPUs to ensure the model captures the maximum possible nuance in facial micro-expressions."* (This sounds very professional).
