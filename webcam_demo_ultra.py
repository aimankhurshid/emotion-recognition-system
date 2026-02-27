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
<<<<<<< HEAD
# CONFIGURATION
# ============================================================================
CAMERA_INDEX = 0  # Default camera
=======
# CONFIGURATION & DISCOVERY
# ============================================================================
def find_cameras():
    available = []
    print("🔍 Probing for cameras (0-4)...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default='0', help='Camera index or URL')
args = parser.parse_args()

# Try to parse numeric index, else use string
try:
    CAMERA_INDEX = int(args.camera)
except:
    CAMERA_INDEX = args.camera

>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
CAMERA_WIDTH = 1280  
CAMERA_HEIGHT = 720
TARGET_FPS = 30  
DETECTION_SCALE = 0.5  
<<<<<<< HEAD
PROCESS_EVERY_N_FRAMES = 8  # Process emotion every 8 frames (every ~0.25s)
DETECT_EVERY_N_FRAMES = 4  # Detect faces every 4 frames
=======
PROCESS_EVERY_N_FRAMES = 3  # Faster updates
DETECT_EVERY_N_FRAMES = 2   # Faster detection
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
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
<<<<<<< HEAD
model = get_model(model_type='full', num_classes=8, pretrained=False, backbone='efficientnet_b4', lstm_hidden=512)
checkpoint_path = r"results\phase1_laptop_benchmark\best_model_full_efficientnet_b4_20260221_004807.pth"
=======
model = get_model(model_type='full', num_classes=8, pretrained=False, backbone='resnet50', lstm_hidden=256, use_projection=False)
checkpoint_path = r"results\demo_checkpoint\best_model_full_resnet50_20260213_122558.pth"
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
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
<<<<<<< HEAD
        pred_idx = output.argmax(1).item()
        
        emotion = emotions[pred_idx]
        confidence = probs[pred_idx].item()
        return emotion, confidence
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return 'Neutral', 0.5
=======
        
        # SENSITIVITY CALIBRATION: Reduce "Neutral" dominance
        boosted_probs = probs.clone()
        
        # 1. Heavily suppress Neutral (index 0) if other emotions have any signal
        boosted_probs[0] *= 0.5
        
        # 2. Boost core expressions (Happy, Surprise, Angry, Sad)
        # EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        boosted_probs[1] *= 2.0  # Happy boost
        boosted_probs[2] *= 1.5  # Sad boost
        boosted_probs[3] *= 2.0  # Surprise boost
        boosted_probs[6] *= 1.5  # Anger boost
        
        pred_idx = boosted_probs.argmax().item()
        
        emotion = emotions[pred_idx]
        confidence = probs[pred_idx].item()
        
        # Return all probabilities for the dashboard
        all_probs = {emotions[i]: probs[i].item() for i in range(len(emotions))}
        return emotion, confidence, all_probs
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return 'Neutral', 0.5, {em: 0.0 for em in emotions}
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def main():
    global latest_frame, should_exit
    
<<<<<<< HEAD
=======
    # 1. Show available cameras first
    cams = find_cameras()
    print(f"✅ ACCESSIBLE CAMERAS: {cams}")
    print(f"🎬 USING CAMERA: {CAMERA_INDEX}")
    if not cams and isinstance(CAMERA_INDEX, int):
        print("⚠️ No local cameras detected. If using DroidCam IP, ensure URL is correct.")
    
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
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
<<<<<<< HEAD
=======
    last_probs = {}
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
    
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
        
<<<<<<< HEAD
=======
        # Dashboard Background (Right Side)
        dash_w = 320
        cv2.rectangle(frame, (frame.shape[1]-dash_w, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
        cv2.putText(frame, "EMOTION LOGIC ENGINE", (frame.shape[1]-dash_w+20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
        # Face detection (throttled)
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            # Aggressive downsampling
            small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
<<<<<<< HEAD
            # Ultra-fast detection
            faces_small = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.4,  # Very fast
                minNeighbors=2,   # Very fast (less accuracy, more speed)
                minSize=(15, 15),
=======
            # Higher quality detection
            faces_small = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More accurate
                minNeighbors=5,   # More stable
                minSize=(30, 30),
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back
            scale = 1.0 / DETECTION_SCALE
            last_faces = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) 
                         for (x, y, w, h) in faces_small]
        
        # Process faces
        for face_idx, (x, y, w, h) in enumerate(last_faces):
<<<<<<< HEAD
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
=======
            # 1. PADDED CROP: Add 25% margin to see eyebrows/chin better
            pad_h = int(h * 0.25)
            pad_w = int(w * 0.25)
            
            y1, y2 = max(0, y - pad_h), min(frame.shape[0], y + h + pad_h)
            x1, x2 = max(0, x - pad_w), min(frame.shape[1], x + w + pad_w)
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            
            # 2. CONTRAST ENHANCEMENT (CLAHE): Helps with poor webcam lighting
            try:
                lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                face_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            except:
                pass # Fallback to raw if CLAHE fails
            
            # Emotion prediction (throttled)
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                emotion, conf, probs_dict = predict_emotion_fast(face_img)
                # TEMPORAL SMOOTHING: Blend with last probabilities to prevent flickering
                if face_idx in last_probs:
                    alpha = 0.6 # 60% new frame, 40% history
                    for k in probs_dict:
                        probs_dict[k] = alpha * probs_dict[k] + (1-alpha) * last_probs[face_idx].get(k, 0)
                    
                    # Re-calculate best after smoothing
                    # Suppress Neutral again in the smoothed pool
                    smooth_boosted = {k: v * (0.5 if k == 'Neutral' else 1.0) for k, v in probs_dict.items()}
                    # Specific boosts for demo
                    smooth_boosted['Happy'] *= 2.0
                    smooth_boosted['Surprise'] *= 2.5
                    smooth_boosted['Anger'] *= 1.5
                    
                    emotion = max(smooth_boosted, key=smooth_boosted.get)
                    conf = probs_dict[emotion]
                
                last_emotions[face_idx] = (emotion, conf)
                last_probs[face_idx] = probs_dict
            else:
                emotion, conf = last_emotions.get(face_idx, ('Neutral', 0.5))
                probs_dict = last_probs.get(face_idx, {em: 0.0 for em in emotions})
            
            # Draw Probabilities on Dashboard (for first face)
            if face_idx == 0:
                for i, (em_name, em_prob) in enumerate(probs_dict.items()):
                    y_pos = 120 + (i * 45)
                    bar_max_w = 200
                    current_bar_w = int(bar_max_w * em_prob)
                    
                    # Highlight active or high-confidence emotions
                    is_current = (em_name == emotion)
                    t_color = emotion_colors.get(em_name, (200, 200, 200))
                    
                    # If very low probability, dim it
                    display_color = t_color if (em_prob > 0.1 or is_current) else (60, 60, 60)
                    
                    cv2.putText(frame, em_name, (frame.shape[1]-dash_w+15, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2 if is_current else 1)
                    
                    # Bar
                    cv2.rectangle(frame, (frame.shape[1]-220, y_pos-12), (frame.shape[1]-20, y_pos+3), (40, 40, 40), -1)
                    if em_prob > 0:
                        cv2.rectangle(frame, (frame.shape[1]-220, y_pos-12), (frame.shape[1]-220+current_bar_w, y_pos+3), t_color, -1)
                    
                    # Percentage
                    cv2.putText(frame, f"{int(em_prob*100)}%", (frame.shape[1]-55, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
>>>>>>> b84b3a84a7bc57632fbec1c421171f80e3049861
            
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
