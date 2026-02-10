#!/bin/zsh
# Install all dependencies and run demo for Wednesday

echo "Installing Python dependencies..."
pip3 install --break-system-packages torch torchvision numpy pillow opencv-python scikit-learn matplotlib seaborn tensorboard tqdm

echo "\n✅ Dependencies installed!"
echo "\nGenerating demo outputs..."

# Navigate to SSD
cd /Volumes/AimanTB/minorproject_facial/emotion_recognition_system

# Create mock data
python3 create_simple_mock_data.py

# Run quick training
python3 training/train.py --epochs 3 --batch_size 8

echo "\n======================================"
echo "✅ DEMO COMPLETE!"  
echo "======================================"
echo "Check results in:"
echo "  - results/checkpoints/confusion_matrix.png"
echo "  - results/checkpoints/training_history_*.png"
echo "\nFor webcam demo, run:"
echo "  python3 webcam_demo_professor.py"
echo "======================================"
