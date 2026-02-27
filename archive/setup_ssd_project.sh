#!/bin/zsh
# Complete SSD Setup Script for Minor Project
# Run this to migrate everything to /Volumes/AimanTB

set -e  # Exit on error

echo "======================================================"
echo "  MINOR PROJECT SSD SETUP"
echo "======================================================"

# Define paths
SSD_PATH="/Volumes/AimanTB/minorproject_facial/emotion_recognition_system"
CURRENT_PATH="/Users/ayeman/Downloads/minorproject_facial/emotion_recognition_system"

# Step 1: Create SSD directory structure
echo "\n[1/5] Creating directory structure on SSD..."
mkdir -p "$SSD_PATH"/{data,results/{checkpoints,logs},models,utils,training}

# Step 2: Copy project files
echo "\n[2/5] Copying project files..."
rsync -av --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
  "$CURRENT_PATH/" "$SSD_PATH/"

# Step 3: Create environment setup
echo "\n[3/5] Creating Python environment info..."
cat > "$SSD_PATH/SETUP_INSTRUCTIONS.md" << 'EOF'
# Setup Instructions

## 1. Install Dependencies
```bash
pip3 install torch torchvision torchaudio
pip3 install opencv-python pillow numpy scikit-learn matplotlib seaborn tensorboard
pip3 install kagglehub
```

## 2. Download Dataset
Run from this directory:
```bash
python3 download_data.py
```

## 3. Train Model
```bash
python3 training/train.py --epochs 30 --batch_size 16 --use_class_weights
```

## 4. Run Demo
```bash
python3 webcam_demo_professor.py
```

## 5. Generate Outputs for Wednesday
```bash
# Quick test (mock data)
python3 create_simple_mock_data.py
python3 training/train.py --epochs 3

# This generates:
# - results/checkpoints/confusion_matrix.png
# - results/checkpoints/training_history_*.png
# - Precision/Recall table in terminal (copy for report)
```
EOF

# Step 4: Create quick run script
echo "\n[4/5] Creating quick test script..."
cat > "$SSD_PATH/quick_demo.sh" << 'EOF'
#!/bin/zsh
# Quick demo for Wednesday presentation

echo "Generating mock data..."
python3 create_simple_mock_data.py

echo "\nRunning 3-epoch training..."
python3 training/train.py --epochs 3 --batch_size 8

echo "\n==================================="
echo "  Demo complete! Check results in:"
echo "  - results/checkpoints/"
echo "===================================" 
EOF

chmod +x "$SSD_PATH/quick_demo.sh"

# Step 5: Verify
echo "\n[5/5] Verifying setup..."
if [ -f "$SSD_PATH/training/train.py" ]; then
    echo "✅ Project files copied successfully"
else
    echo "❌ Error: Project files not found"
    exit 1
fi

echo "\n======================================================"
echo "  ✅ SETUP COMPLETE!"
echo "======================================================"
echo "Next steps:"
echo "1. cd $SSD_PATH"
echo "2. Read SETUP_INSTRUCTIONS.md"
echo "3. Run: ./quick_demo.sh (for immediate output)"
echo "======================================================"
