#!/bin/bash
# RAF-DB Training Setup Script for RTX 5000 Ada
# Run this at university PC at 9:30 AM

echo "=== RAF-DB Training on RTX 5000 Ada Setup ==="
echo "Start time: $(date)"

# 1. Setup environment
echo "[1/6] Creating virtual environment..."
python3 -m venv uni_env
source uni_env/bin/activate

# 2. Install dependencies
echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install PyTorch with CUDA 12.1
echo "[3/6] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verify GPU
echo "[4/6] Verifying GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# 5. Download RAF-DB (if not already present)
echo "[5/6] Checking RAF-DB dataset..."
if [ ! -d "data/train/0_neutral" ]; then
    echo "Dataset not found. Downloading RAF-DB from Kaggle..."
    pip install kaggle
    kaggle datasets download -d shuvoalok/raf-db-dataset
    unzip -q raf-db-dataset.zip -d data/
    rm raf-db-dataset.zip
    echo "Dataset downloaded successfully!"
else
    echo "Dataset already exists. Verifying..."
    echo "Train images: $(find data/train -type f | wc -l)"
    echo "Val images: $(find data/val -type f | wc -l)"
fi

# 6. Start training
echo "[6/6] Starting RAF-DB training on RTX 5000 Ada..."
echo "This will take approximately 1-2 hours"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

python training/train_rtx5000.py

echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Check results in: results/demo_checkpoint/"
