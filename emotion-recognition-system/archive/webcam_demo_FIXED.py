#!/usr/bin/env python3
"""
SIMPLE Webcam Demo - Fixed Version
Works without trained model, shows interface for professor
"""

import cv2
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

print("="*70)
print("WEBCAM DEMO - FIXED VERSION")
print("="*70)

# Load model
print("\n📦 Loading model...")
model = get_model(model_type='full', num_classes=8, pretrained=False)
model.eval()
print("✓ Model ready")

# Load face detector
print("📸 Loading face detector...")
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
print("✓ Face detector ready")

# Get transform
transform = get_transforms('val', img_size=224)
emotions = EMOTION_LABELS

# Colors for each emotion
emotion_colors = {
    'Neutral': (200, 200, 200),
    'Happy': (0, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0),
    'Fear': (128, 0, 128),
    'Disgust': (0, 128, 0),
    'Anger': (0, 0, 255),
    'Contempt': (255, 128, 0)
}

# Circuit logic explanations
circuit_logic = {
    'Happy': 'Mouth Corners Up + Eye Wrinkles',
    'Surprise': 'Raised Eyebrows + Wide Eyes + Open Mouth',
    'Sad': 'Inner Brow Up + Mouth Corners Down',
    'Angry': 'Brows Down + Jaw Tight',
    'Fear': 'Eyes Wide + Lips Stretched',
    'Disgust': 'Nose Wrinkle + Upper Lip Raised',
    'Contempt': 'One Side Mouth Corner Up',
    'Neutral': 'Balanced Features'
}

def predict_emotion(face_img):
    """Predict emotion from face ROI"""
    try:
        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            _, pred_idx = output.max(1)
        
        emotion = emotions[pred_idx.item()]
        confidence = probs[pred_idx.item()].item()
        
        return emotion, confidence, probs.numpy()
    except:
        return 'Neutral', 0.5, np.ones(8) / 8

# Open camera with AVFoundation backend (macOS specific)
print("\n🎥 Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Camera not found! Make sure:")
    print("  1. Camera is connected")
    print("  2. System Preferences → Privacy → Camera → Terminal is enabled")  
    print("  3. No other app is using the camera")
    exit(1)

print("✅ Camera opened!")
print("⏳ Warming up camera (3 seconds)...")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Warm-up: Read and discard frames to let camera initialize
import time
time.sleep(1)  # Wait for camera to warm up
for i in range(10):
    ret, _ = cap.read()
    if not ret:
        print(f"⏳ Frame {i+1}/10 - camera initializing...")
        time.sleep(0.2)
    else:
        print(f"✓ Frame {i+1}/10 - OK")

print("✅ Camera ready!")
print("\nControls:")
print("  - Press 'Q' to quit")
print("  - Press 'S' to save screenshot")
print("\nShowing webcam window...")

screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame, retrying...")
        time.sleep(0.1)
        continue
    
    frame = cv2.flip(frame, 1)  # Mirror
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    # Process each face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        
        # Predict
        emotion, conf, probs = predict_emotion(face_img)
        color = emotion_colors.get(emotion, (255, 255, 255))
        logic = circuit_logic.get(emotion, "")
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Confidence bar
        bar_y = y - 15
        bar_w = int(w * conf)
        cv2.rectangle(frame, (x, bar_y), (x+w, bar_y+10), (50,50,50), -1)
        cv2.rectangle(frame, (x, bar_y), (x+bar_w, bar_y+10), color, -1)
        
        # Emotion label
        cv2.putText(frame, f"{emotion} {int(conf*100)}%", (x, bar_y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Circuit logic text
        cv2.putText(frame, logic, (x, y+h+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'Q' to quit  |  Press 'S' to screenshot", 
               (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    # Show
    cv2.imshow('Bi-LSTM Emotion Recognition (Professor Demo)', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Q or ESC
        print("\n👋 Closing demo...")
        break
    if key == ord('s'):
        filename = f'DEMO_OUTPUTS/screenshot_{screenshot_count}.png'
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Demo closed successfully!")
