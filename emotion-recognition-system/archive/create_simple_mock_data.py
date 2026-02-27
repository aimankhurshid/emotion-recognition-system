import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_mock_dataset(root_dir="data"):
    """Creates mock dataset with PIL (no cv2 needed)"""
    
    emotions = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
        4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'
    }
    
    splits = ['train', 'val', 'test']
    
    print(f"Creating mock dataset in '{root_dir}'...")
    
    for split in splits:
        for idx, emotion in emotions.items():
            folder_name = f"{idx}_{emotion}"
            dir_path = os.path.join(root_dir, split, folder_name)
            os.makedirs(dir_path, exist_ok=True)
            
            num_images = 10 if split == 'train' else 3
            
            for i in range(num_images):
                # Create random RGB image
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                
                # Add text
                draw = ImageDraw.Draw(img)
                draw.text((80, 100), emotion, fill=(255, 255, 255))
                
                filename = os.path.join(dir_path, f"mock_{i}.jpg")
                img.save(filename)
    
    print("✅ Mock dataset created successfully!")
    print(f"Run 'python training/train.py --epochs 3' to test.")

if __name__ == "__main__":
    create_mock_dataset()
