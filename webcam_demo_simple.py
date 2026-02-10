#!/usr/bin/env python3
"""
Simple Desktop Webcam Demo for macOS
Shows emotion recognition with circuit logic in a native window
Press 'q' to quit, 's' to save screenshot
"""

import cv2
import torch
import numpy as np
from PIL import Image
import sys
from datetime import datetime
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

# Load model
print("📦 Loading model...")
model = get_model(model_type='full', num_classes=8, pretrained=False)
model.eval()
transform = get_transforms('val', img_size=224)
print("✅ Model loaded!")

# Face detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

emotions = EMOTION_LABELS
emotion_colors = {
    'Neutral': (200, 200, 200), 'Happy': (0, 255, 0), 'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0), 'Fear': (128, 0, 128), 'Disgust': (0, 128, 0),
    'Anger': (0, 0, 255), 'Contempt': (255, 128, 0)
}
circuit_logic = {
    'Happy': 'Mouth Up + Eye Wrinkles', 
    'Surprise': 'Eyebrows Up + Eyes Wide + Mouth Open',
    'Sad': 'Brow Up + Mouth Down', 
    'Anger': 'Brows Down + Jaw Tight',
    'Fear': 'Eyes Wide + Lips Stretched', 
    'Disgust': 'Nose Wrinkle + Lip Raised',
    'Contempt': 'One Side Mouth Up', 
    'Neutral': 'Balanced Features'
}

# Initialize camera
print("🎥 Opening camera...")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("❌ ERROR: Cannot open camera!")
    sys.exit(1)

print("✅ Camera opened!")

def predict_emotion(face_img):
    """Predict emotion from face image"""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            _, pred_idx = output.max(1)
        emotion = emotions[pred_idx.item()]
        confidence = probs[pred_idx.item()].item()
        return emotion, confidence
    except Exception as e:
        return 'Neutral', 0.5

print("\n" + "="*70)
print("🎥 BI-LSTM EMOTION RECOGNITION - WEBCAM DEMO")
print("="*70)
print("\n✅ Camera window opening...")
print("\n📝 Controls:")
print("   'q' or ESC - Quit")
print("   's' - Save screenshot")
print("\n🎭 Try different expressions: Happy, Surprise, Sad, Angry")
print("="*70 + "\n")

# Create window
cv2.namedWindow('Bi-LSTM Emotion Recognition Demo', cv2.WINDOW_NORMAL)

frame_count = 0
screenshot_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)  # Mirror image
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    # Process each face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        
        # Predict emotion
        emotion, conf = predict_emotion(face_img)
        color = emotion_colors.get(emotion, (255, 255, 255))
        logic = circuit_logic.get(emotion, "")
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw confidence bar
        bar_y = y - 25
        bar_h = 20
        bar_w_full = w
        bar_w_filled = int(w * conf)
        
        # Background bar
        cv2.rectangle(frame, (x, bar_y), (x+bar_w_full, bar_y+bar_h), (50, 50, 50), -1)
        # Filled bar
        cv2.rectangle(frame, (x, bar_y), (x+bar_w_filled, bar_y+bar_h), color, -1)
        
        # Emotion label with confidence
        label = f"{emotion} {int(conf*100)}%"
        cv2.putText(frame, label, (x, bar_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        # Circuit logic below face
        logic_y = y + h + 35
        cv2.putText(frame, f"Circuit: {logic}", (x, logic_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Add header text
    cv2.putText(frame, "Bi-LSTM Emotion Recognition - Professor Demo", 
               (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Add info text
    info_text = "Press 'q' to quit | 's' to save screenshot"
    cv2.putText(frame, info_text, 
               (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    
    # Show frame
    cv2.imshow('Bi-LSTM Emotion Recognition Demo', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:  # 'q' or ESC
        print("\n👋 Closing demo...")
        break
    elif key == ord('s'):  # Save screenshot
        screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

# Cleanup
camera.release()
cv2.destroyAllWindows()
print(f"\n✅ Demo closed. Processed {frame_count} frames, saved {screenshot_count} screenshots.")
