#!/usr/bin/env python3
"""
Setup script for downloading and verifying the AffectNet+ dataset
"""

import os
import sys
import json
from pathlib import Path

def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This will download kaggle.json")
        print(f"4. Move it to: {kaggle_json}")
        print(f"5. Run: chmod 600 {kaggle_json}")
        return False
    
    print(f"✅ Kaggle credentials found at {kaggle_json}")
    return True


def download_dataset():
    """Download AffectNet+ dataset from Kaggle"""
    try:
        import kaggle
        
        dataset_name = "dollyprajapati182/balanced-affectnet"
        download_path = "data"
        
        print(f"\n📥 Downloading dataset: {dataset_name}")
        print(f"Destination: {download_path}/")
        print("This may take 10-30 minutes depending on your connection...")
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True,
            quiet=False
        )
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet")
        print("2. Click 'Download' button")
        print("3. Extract the zip file to the 'data/' directory")
        return False


def verify_dataset_structure():
    """Verify the dataset is properly structured"""
    data_dir = Path('data')
    
    required_splits = ['train', 'val', 'test']
    emotions = ['0_neutral', '1_happy', '2_sad', '3_surprise', 
                '4_fear', '5_disgust', '6_anger', '7_contempt']
    
    print("\n🔍 Verifying dataset structure...")
    
    all_good = True
    total_images = 0
    
    for split in required_splits:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"❌ Missing: {split_dir}")
            all_good = False
            continue
        
        print(f"\n{split.upper()} SET:")
        split_total = 0
        
        for emotion in emotions:
            emotion_dir = split_dir / emotion
            
            if not emotion_dir.exists():
                # Try alternative naming (just numbers)
                emotion_dir = split_dir / emotion.split('_')[0]
            
            if emotion_dir.exists():
                images = list(emotion_dir.glob('*.jpg')) + \
                        list(emotion_dir.glob('*.png')) + \
                        list(emotion_dir.glob('*.jpeg'))
                count = len(images)
                split_total += count
                print(f"  {emotion}: {count} images")
            else:
                print(f"  {emotion}: NOT FOUND ❌")
                all_good = False
        
        print(f"  Total: {split_total} images")
        total_images += split_total
    
    print(f"\n{'='*50}")
    print(f"TOTAL IMAGES IN DATASET: {total_images}")
    print(f"{'='*50}")
    
    if all_good and total_images > 0:
        print("✅ Dataset structure verified!")
        return True
    else:
        print("⚠️ Dataset structure issues detected")
        return False


def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("\n🎨 Creating sample dataset for testing...")
    
    import numpy as np
    from PIL import Image
    
    data_dir = Path('data')
    emotions = ['0_neutral', '1_happy', '2_sad', '3_surprise', 
                '4_fear', '5_disgust', '6_anger', '7_contempt']
    
    for split in ['train', 'val', 'test']:
        num_samples = 10 if split == 'train' else 3
        
        for emotion in emotions:
            emotion_dir = data_dir / split / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_samples):
                # Create random colored image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(emotion_dir / f"sample_{i:03d}.jpg")
    
    print("✅ Sample dataset created!")
    print("Note: This is synthetic data for testing only")
    return True


def main():
    """Main setup function"""
    print("="*60)
    print("AffectNet+ Dataset Setup")
    print("="*60)
    
    # Check for existing dataset
    if verify_dataset_structure():
        print("\n✅ Dataset already set up and verified!")
        return
    
    print("\nChoose an option:")
    print("1. Download AffectNet+ from Kaggle (requires credentials)")
    print("2. Use sample synthetic dataset (for testing code)")
    print("3. Skip (I'll download manually)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        if check_kaggle_credentials():
            if download_dataset():
                verify_dataset_structure()
        else:
            print("\nPlease set up Kaggle credentials and run this script again")
    
    elif choice == '2':
        create_sample_dataset()
        print("\n⚠️ Remember: Replace with real data before final training!")
    
    elif choice == '3':
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet")
        print("2. Download and extract to 'data/' directory")
        print("3. Run this script again to verify")
    
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
