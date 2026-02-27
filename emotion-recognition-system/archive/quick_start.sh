#!/bin/bash

# Quick start script for emotion recognition system
# This script sets up the environment and provides menu-driven options

set -e

echo "==========================================="
echo "Emotion Recognition System - Quick Start"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check for dataset
if [ ! -d "data/train" ]; then
    echo ""
    echo "⚠️  Dataset not found in data/ directory"
    echo "Please download AffectNet+ dataset from:"
    echo "https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet"
    echo ""
    read -p "Press Enter to continue without dataset (for code review only)..."
fi

# Menu
while true; do
    echo ""
    echo "==========================================="
    echo "What would you like to do?"
    echo "==========================================="
    echo "1. Train model (full 50 epochs)"
    echo "2. Train model (quick demo - 5 epochs)"
    echo "3. Evaluate model"
    echo "4. Run ablation study"
    echo "5. Test single image prediction"
    echo "6. Run webcam demo"
    echo "7. Open Jupyter notebook"
    echo "8. Check model architecture"
    echo "9. Exit"
    echo ""
    read -p "Enter your choice (1-9): " choice

    case $choice in
        1)
            echo "Starting full training (50 epochs)..."
            cd training
            python train.py --data_dir ../data --epochs 50 --batch_size 32
            cd ..
            ;;
        2)
            echo "Starting demo training (5 epochs)..."
            cd training
            python train.py --data_dir ../data --epochs 5 --batch_size 16
            cd ..
            ;;
        3)
            echo "Enter path to model checkpoint:"
            read -p "Path: " checkpoint_path
            cd training
            python evaluate.py --checkpoint_path "$checkpoint_path" --data_dir ../data
            cd ..
            ;;
        4)
            echo "Running ablation study (may take several hours)..."
            cd training
            python ablation_study.py --data_dir ../data --epochs 20
            cd ..
            ;;
        5)
            echo "Enter path to image:"
            read -p "Image path: " image_path
            echo "Enter path to model checkpoint:"
            read -p "Checkpoint path: " checkpoint_path
            cd inference
            python predict_single.py --image_path "$image_path" --model_path "$checkpoint_path" --visualize
            cd ..
            ;;
        6)
            echo "Enter path to model checkpoint:"
            read -p "Checkpoint path: " checkpoint_path
            cd inference
            python webcam_demo.py --model_path "$checkpoint_path"
            cd ..
            ;;
        7)
            echo "Opening Jupyter notebook..."
            jupyter notebook notebooks/full_pipeline.ipynb
            ;;
        8)
            echo "Checking model architecture..."
            python -c "
import sys
sys.path.append('.')
from models import get_model, count_parameters

model = get_model(model_type='full', num_classes=8, backbone='efficientnet_b4')
print(model)
count_parameters(model)
"
            ;;
        9)
            echo "Exiting..."
            deactivate
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-9."
            ;;
    esac
done
