# Remote Training Setup Guide

This guide is for setting up the emotion recognition system on a remote machine (e.g., a friend's gaming laptop with NVIDIA GPU).

## 1. Prerequisites

- **Git**: Ensure Git is installed.
- **Python**: Install Python 3.9 or 3.10.
- **NVIDIA Drivers**: Ensure the latest NVIDIA Game Ready or Studio Drivers are installed.
- **Sudo/Admin Access**: Required for installing some packages.

## 2. Clone the Repository

Open a terminal (PowerShell on Windows or Bash on Linux) and run:

```bash
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion_recognition_system
```

## 3. Setup Virtual Environment

It is highly recommended to use a virtual environment.

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
```

*Note: If you have an NVIDIA GPU, ensure `torch` is installed with CUDA support. You might need to run the following (visit [pytorch.org](https://pytorch.org/) for the exact command for your CUDA version):*

```bash
# Example for CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 5. Prepare Data

If the dataset is not included in the repo (checked out in `.gitignore`), you need to copy the `data/` folder from the source or download it again.

Place the dataset in: `emotion_recognition_system/data/`

## 6. Run Training

To start the training process:

```bash
python training/train.py
```

The training checkpoints will be saved in `results/checkpoints/`.

## 7. Monitoring

You can see the progress in the terminal. If enabled, TensorBoard logs will be in `results/logs/`.
