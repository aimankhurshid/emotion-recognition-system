"""
Data loading utilities for AffectNet+ dataset
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class AffectNetDataset(Dataset):
    """
    PyTorch Dataset for AffectNet+
    
    Expected directory structure:
    data/
    ├── train/
    │   ├── 0_neutral/
    │   ├── 1_happy/
    │   ├── 2_sad/
    │   ├── 3_surprise/
    │   ├── 4_fear/
    │   ├── 5_disgust/
    │   ├── 6_anger/
    │   └── 7_contempt/
    └── annotations.csv (optional)
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: PyTorch transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.emotion_labels = {
            0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise',
            4: 'Fear', 5: 'Disgust', 6: 'Anger', 7: 'Contempt'
        }
        
        self.data = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load image paths and labels"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Creating sample structure...")
            return
        
        for emotion_id in range(8):
            emotion_name = self.emotion_labels[emotion_id]
            class_dir = os.path.join(split_dir, f"{emotion_id}_{emotion_name.lower()}")
            
            if not os.path.exists(class_dir):
                class_dir = os.path.join(split_dir, str(emotion_id))
            
            # Add support for plain text names (e.g., "neutral", "happy") common in Kaggle datasets
            if not os.path.exists(class_dir):
                class_dir = os.path.join(split_dir, emotion_name.lower())
            
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(emotion_id)
        
        print(f"Loaded {len(self.data)} images for {self.split} split")
        
        if len(self.data) > 0:
            unique, counts = np.unique(self.labels, return_counts=True)
            for emotion_id, count in zip(unique, counts):
                print(f"  {self.emotion_labels[emotion_id]}: {count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get data transforms for training/validation/test
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Input image size
    
    Returns:
        torchvision transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def compute_class_weights(dataset, num_classes=8):
    """
    Compute class weights for handling imbalanced dataset
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
    
    Returns:
        Class weights as numpy array
    """
    labels = np.array(dataset.labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    
    print("Class weights:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}")
    
    return class_weights


def split_dataset(data_dir, val_split=0.15, test_split=0.15, random_state=42):
    """
    Create train/val/test splits if not already separated
    
    Args:
        data_dir: Directory containing all images
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed
    
    Returns:
        train_files, val_files, test_files
    """
    all_files = []
    all_labels = []
    
    for emotion_id in range(8):
        emotion_dir = os.path.join(data_dir, str(emotion_id))
        if os.path.exists(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(emotion_dir, img_name))
                    all_labels.append(emotion_id)
    
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, 
        test_size=(val_split + test_split), 
        random_state=random_state,
        stratify=all_labels
    )
    
    val_ratio = val_split / (val_split + test_split)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    return train_files, val_files, test_files


def get_data_loaders(data_dir, batch_size=32, num_workers=4, img_size=224, persistent_workers=True):
    """
    Create PyTorch data loaders for train/val/test
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Input image size
        persistent_workers: Keep workers alive between epochs (faster)
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('val', img_size)
    
    train_dataset = AffectNetDataset(data_dir, split='train', transform=train_transform)
    val_dataset = AffectNetDataset(data_dir, split='val', transform=val_transform)
    test_dataset = AffectNetDataset(data_dir, split='test', transform=val_transform)
    
    class_weights = None
    if len(train_dataset) > 0:
        class_weights = compute_class_weights(train_dataset)
    
    # Optimized dataloader kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader, class_weights
