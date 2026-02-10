# Laptop Setup Guide (Using Git)

Since you downloaded the project via Git, setting up should be fast!

## 1. Clone the Project

On your friend's laptop (Windows recommended for gaming laptops), open a terminal (PowerShell or Git Bash) and run:

```bash
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion_recognition_system
```

## 2. Check Your Dataset (Important!)

The repository includes a **small sample dataset (~8MB)** in the `data/` folder.
- If you want to train on this sample, skip to **Step 3**.
- **If you have a larger full dataset (GBs)** on a USB drive or cloud:
    1.  Delete the existing `data` folder:
        *   Windows: `rd /s /q data`
        *   Mac/Linux: `rm -rf data`
    2.  Copy your **full** `data` folder into this directory so it replaces the small one.

## 3. Setup Environment

It's best to use a fresh virtual environment.

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** If you have an NVIDIA GPU, install the specific PyTorch version for CUDA:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
*(Example: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)*

## 4. Run Training

Now you can start training!

```bash
python training/train.py
```

The model checkpoints will be saved in `results/checkpoints/`.

## 5. Run Demo

To test the webcam with your new model (or the pre-trained one if you downloaded it):

```bash
python webcam_demo_ultra.py
```
