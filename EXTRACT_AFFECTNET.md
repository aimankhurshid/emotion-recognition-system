# How to Extract AffectNet+ Multi-Part Archives

Your 5-part ZIP files (totaling 7.6GB - 8GB) are the **Official University of Denver (DU) Release**. macOS's built-in `unzip` can't handle them properly.

## ✅ Solution: Use 7-Zip or The Unarchiver

### Option 1: Install The Unarchiver (Easiest for Mac)
1. Download from: https://theunarchiver.com/
2. Install it
3. Right-click `AffectNet+_part1.zip` → Open With → The Unarchiver
4. It will automatically detect and extract all 4 parts

### Option 2: Use Homebrew + 7z (Terminal Method)
```bash
# Install 7z if you don't have it
brew install p7zip

# Navigate to your SSD
cd /Volumes/AimanTB/affectnet+/

# Extract the multi-part archive (7z will auto-detect all parts)
7z x AffectNet+_part1.zip
```

### Option 3: Manual Concatenation (What You Tried - But Correctly)
The issue with your `cat` command was that you overwrote the original `AffectNet+.zip`. 

**Delete the wrong one first:**
```bash
cd /Volumes/AimanTB/affectnet+/
rm AffectNet+.zip  # Remove the incorrectly created one
rm -rf AffectNet+  # Remove the wrong extraction
```

**Then combine them properly using zip's built-in multi-part support:**
```bash
zip -s 0 AffectNet+_part1.zip --out complete_affectnet.zip
unzip complete_affectnet.zip
```

---

## 🎯 Expected Result

After successful extraction, you should see:
```
AffectNet+/
├── train/
│   ├── 0_neutral/
│   ├── 1_happy/
│   ├── 2_sad/
│   ├── 3_surprise/
│   ├── 4_fear/
│   ├── 5_disgust/
│   ├── 6_anger/
│   └── 7_contempt/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

**Total size**: ~100-120GB

---

## ⚠️ Important Note

The `no_human_annotated` folder you keep extracting is from the main `AffectNet+.zip` (7.4GB single file). That one is NOT what you need. You need the **part1-4 series**.
