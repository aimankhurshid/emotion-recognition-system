#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Desktop Webcam Demo
Multi-threaded architecture for maximum performance
Target: 60+ FPS on CPU, 120+ FPS on GPU
"""

import cv2
import torch
import numpy as np
from PIL import Image
import sys
import time
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

# ============================================================================
# CONFIGURATION
# ============================================================================
CAMERA_INDEX = 1  # Mac built-in camera
CAMERA_WIDTH = 640  # Lower resolution for max FPS
CAMERA_HEIGHT = 360
TARGET_FPS = 30  # Realistic target for CPU
DETECTION_SCALE = 0.3  # Very aggressive downscaling
PROCESS_EVERY_N_FRAMES = 8  # Process emotion every 8 frames (every ~0.25s)
DETECT_EVERY_N_FRAMES = 4  # Detect faces every 4 frames
USE_THREADING = True
MAX_QUEUE_SIZE = 2  # Small queue to prevent lag

# ============================================================================
# GLOBAL STATE
# ============================================================================
frame_lock = Lock()
latest_frame = None
should_exit = False

# Model setup
print("📦 Loading model...")
model = get_model(model_type='full', num_classes=8, pretrained=False)
model.eval()

# GPU optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if device.type == 'cuda':
    print("🚀 GPU acceleration enabled!")
    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
else:
    print("💻 Running on CPU")

transform = get_transforms('val', img_size=224)
print("✅ Model loaded!")

# Face detector - optimized
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

# ============================================================================
# CAMERA CAPTURE THREAD
# ============================================================================
class CameraThread(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.camera = None
        
    def run(self):
        global latest_frame, should_exit
        
        print("🎥 Opening camera...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        # Ultra-optimized camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 60)  # Max FPS
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG codec for faster capture
        
        if not self.camera.isOpened():
            print("❌ ERROR: Cannot open camera!")
            should_exit = True
            return
            
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        print(f"✅ Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        while not should_exit:
            ret, frame = self.camera.read()
            if ret:
                with frame_lock:
                    latest_frame = cv2.flip(frame, 1)  # Mirror
            else:
                break
                
        self.camera.release()
        
    def stop(self):
        global should_exit
        should_exit = True

# ============================================================================
# OPTIMIZED EMOTION PREDICTION
# ============================================================================
@torch.no_grad()  # Disable gradient computation for inference
def predict_emotion_fast(face_img):
    """Ultra-fast emotion prediction with caching"""
    try:
        # Convert and transform
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()
        
        emotion = emotions[pred_idx]
        confidence = probs[pred_idx].item()
        return emotion, confidence
    except Exception as e:
        return 'Neutral', 0.5

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def main():
    global latest_frame, should_exit
    
    print("\n" + "="*70)
    print("🚀 ULTRA-OPTIMIZED BI-LSTM EMOTION RECOGNITION")
    print("="*70)
    print("🎯 Target: 60+ FPS | Multi-threaded Processing")
    print("\n📝 Controls:")
    print("   'q' or ESC - Quit")
    print("   's' - Save screenshot")
    print("="*70 + "\n")
    
    # Start camera thread
    if USE_THREADING:
        camera_thread = CameraThread()
        camera_thread.start()
        time.sleep(1)  # Wait for camera initialization
    else:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Create window
    cv2.namedWindow('Ultra-Fast Emotion Recognition', cv2.WINDOW_NORMAL)
    
    # Performance tracking
    frame_count = 0
    screenshot_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    # Caching
    last_faces = []
    last_emotions = {}
    
    # Main loop
    while not should_exit:
        # Get frame
        if USE_THREADING:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
        else:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
        
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Face detection (throttled)
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            # Aggressive downsampling
            small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            # Ultra-fast detection
            faces_small = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.4,  # Very fast
                minNeighbors=2,   # Very fast (less accuracy, more speed)
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back
            scale = 1.0 / DETECTION_SCALE
            last_faces = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) 
                         for (x, y, w, h) in faces_small]
        
        # Process faces
        for face_idx, (x, y, w, h) in enumerate(last_faces):
            # Extract face ROI
            face_img = frame[max(0,y):y+h, max(0,x):x+w]
            if face_img.size == 0:
                continue
            
            # Emotion prediction (throttled)
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                emotion, conf = predict_emotion_fast(face_img)
                last_emotions[face_idx] = (emotion, conf)
            else:
                emotion, conf = last_emotions.get(face_idx, ('Neutral', 0.5))
            
            color = emotion_colors.get(emotion, (255, 255, 255))
            logic = circuit_logic.get(emotion, "")
            
            # Draw (optimized)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Confidence bar
            bar_y = y - 20
            bar_h = 15
            bar_w = w
            bar_filled = int(w * conf)
            cv2.rectangle(frame, (x, bar_y), (x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, bar_y), (x+bar_filled, bar_y+bar_h), color, -1)
            
            # Labels
            label = f"{emotion} {int(conf*100)}%"
            cv2.putText(frame, label, (x, bar_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, logic, (x, y+h+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Header
        cv2.putText(frame, "Ultra-Fast Emotion Recognition", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # FPS display
        fps_text = f"FPS: {int(current_fps)}"
        cv2.putText(frame, fps_text, 
                   (frame.shape[1] - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Info
        info = "Press 'q' to quit | 's' to save"
        cv2.putText(frame, info, 
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display
        cv2.imshow('Ultra-Fast Emotion Recognition', frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n👋 Closing...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultra_demo_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
    
    # Cleanup
    should_exit = True
    if USE_THREADING:
        camera_thread.join(timeout=2)
    else:
        camera.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Closed. Processed {frame_count} frames, {screenshot_count} screenshots")
    print(f"📊 Average FPS: {int(current_fps)}")

if __name__ == "__main__":
    main()
