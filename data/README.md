# Dataset Setup Guide

This guide will help you download and prepare the AffectNet+ dataset.

## Method 1: Kaggle API (Recommended)

### Step 1: Install Kaggle API
```bash
pip install kaggle
```

### Step 2: Set up Kaggle credentials

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` file to:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<Username>\.kaggle\kaggle.json`

4. Set permissions (Linux/Mac only):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download AffectNet+
```bash
cd emotion_recognition_system
kaggle datasets download -d dollyprajapati182/balanced-affectnet
unzip balanced-affectnet.zip -d data/
```

## Method 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet
2. Click "Download" button
3. Extract the zip file
4. Move extracted files to `emotion_recognition_system/data/`

## Expected Directory Structure

After extraction, your directory should look like:

```
emotion_recognition_system/
└── data/
    ├── train/
    │   ├── 0_neutral/
    │   │   ├── image001.jpg
    │   │   ├── image002.jpg
    │   │   └── ...
    │   ├── 1_happy/
    │   ├── 2_sad/
    │   ├── 3_surprise/
    │   ├── 4_fear/
    │   ├── 5_disgust/
    │   ├── 6_anger/
    │   └── 7_contempt/
    ├── val/
    │   └── [same structure as train]
    └── test/
        └── [same structure as train]
```

## Verify Dataset

Run this Python script to verify your dataset:

```python
import os

data_dir = 'data'
emotions = ['0_neutral', '1_happy', '2_sad', '3_surprise', 
            '4_fear', '5_disgust', '6_anger', '7_contempt']

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()} SET:")
    total = 0
    for emotion in emotions:
        path = os.path.join(data_dir, split, emotion)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  {emotion}: {count} images")
            total += count
        else:
            print(f"  {emotion}: NOT FOUND")
    print(f"  Total: {total} images")
```

## Dataset Statistics

**AffectNet+** is a balanced version of AffectNet with:
- **8 emotion classes**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
- **~4000 images per class** (balanced)
- **Resolution**: Vary (will be resized to 224×224)
- **Split**: 70% train / 15% validation / 15% test

## Troubleshooting

### Issue: "403 Forbidden" when downloading
**Solution**: Make sure you're logged into Kaggle and have accepted the dataset terms.

### Issue: Directory structure doesn't match
**Solution**: The dataset might have a different structure. Check the actual structure and modify `utils/data_loader.py` accordingly.

### Issue: Out of disk space
**Solution**: You need ~20GB free space for the dataset and model checkpoints.

## Alternative Dataset

If AffectNet+ is unavailable, you can use:
- **FER2013**: https://www.kaggle.com/datasets/msambare/fer2013
- **RAF-DB**: http://www.whdeng.cn/raf/model1.html
- **AffWild2**: https://ibug.doc.ic.ac.uk/resources/affwild2/

Note: You'll need to modify the data loader for different datasets.
