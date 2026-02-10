"""
Single image emotion prediction
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, load_checkpoint, EMOTION_LABELS
from utils import get_transforms


def detect_face(image_path, face_cascade_path=None):
    """
    Detect face in image using OpenCV
    
    Args:
        image_path: Path to input image
        face_cascade_path: Path to Haar cascade XML (optional)
    
    Returns:
        Face image or None
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if face_cascade_path and os.path.exists(face_cascade_path):
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected. Using full image.")
        return image
    
    x, y, w, h = faces[0]
    
    padding = int(0.2 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    face_image = image[y1:y2, x1:x2]
    
    return face_image


def predict_emotion(model, image, transform, device):
    """
    Predict emotion from image
    
    Args:
        model: Trained model
        image: Input image (numpy array or PIL Image)
        transform: Image transforms
        device: Device to run inference
    
    Returns:
        Predicted emotion label, confidence, all probabilities
    """
    model.eval()
    
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    emotion_id = predicted.item()
    emotion_label = EMOTION_LABELS[emotion_id]
    confidence_score = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    return emotion_label, confidence_score, all_probs


def visualize_prediction(image_path, emotion_label, confidence, all_probs, save_path=None):
    """Visualize prediction with bar chart"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {emotion_label}\nConfidence: {confidence*100:.2f}%', 
                  fontsize=14, fontweight='bold', color='green')
    
    colors = ['#FF6B6B' if i == EMOTION_LABELS.index(emotion_label) else '#4ECDC4' 
              for i in range(len(EMOTION_LABELS))]
    
    bars = ax2.barh(EMOTION_LABELS, all_probs * 100, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, all_probs)):
        ax2.text(prob * 100 + 1, i, f'{prob*100:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found!")
        return
    
    print("\nCreating model...")
    model = get_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False
    )
    model = model.to(device)
    
    print(f"Loading checkpoint from {args.model_path}...")
    load_checkpoint(model, None, args.model_path, device=device)
    
    transform = get_transforms('val', img_size=args.img_size)
    
    print(f"\nProcessing image: {args.image_path}")
    
    if args.detect_face:
        face_image = detect_face(args.image_path)
    else:
        face_image = cv2.imread(args.image_path)
    
    emotion_label, confidence, all_probs = predict_emotion(
        model, face_image, transform, device
    )
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Emotion: {emotion_label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Probabilities:")
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, all_probs)):
        print(f"  {label:<12}: {prob*100:>6.2f}%")
    print("="*60)
    
    if args.visualize:
        save_path = args.output_path if args.output_path else None
        visualize_prediction(args.image_path, emotion_label, confidence, all_probs, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Emotion from Single Image')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization')
    
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'simple_cnn', 'cnn_attention'],
                        help='Model architecture type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        choices=['efficientnet_b4', 'resnet50'],
                        help='CNN backbone')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of emotion classes')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    
    parser.add_argument('--detect_face', action='store_true', default=True,
                        help='Detect and crop face before prediction')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize prediction results')
    
    args = parser.parse_args()
    
    main(args)
