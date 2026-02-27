import os
import cv2
import numpy as np

def create_mock_dataset(root_dir="data"):
    """
    Creates a mock dataset structure with synthetic images to verify the training pipeline.
    Structure matches what utils/data_loader.py expects:
    data/
      ├── train/
      │   ├── 0_neutral/
      │   ├── 1_happy/
      │   ...
      ├── val/
      └── test/
    """
    
    # Emotion labels map to folders
    emotions = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
        4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'
    }
    
    splits = ['train', 'val', 'test']
    
    print(f"Creating mock dataset in '{root_dir}'...")
    
    for split in splits:
        for idx, emotion in emotions.items():
            # Folder name format: "0_neutral" or just "0" depending on loader logic
            # The loader checks for "0_neutral" first, then "0". We'll use "0_neutral"
            folder_name = f"{idx}_{emotion}"
            dir_path = os.path.join(root_dir, split, folder_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create 5 fake images per class per split
            num_images = 5 if split == 'train' else 2
            
            for i in range(num_images):
                # Create a 224x224 RGB random noise image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add some text so we know it's fake
                cv2.putText(img, f"{emotion}", (50, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                filename = os.path.join(dir_path, f"mock_{i}.jpg")
                cv2.imwrite(filename, img)
    
    print("✅ Mock dataset created successfully!")
    print(f"Run 'python training/train.py --epochs 2' to test the code pipeline.")

if __name__ == "__main__":
    create_mock_dataset()
