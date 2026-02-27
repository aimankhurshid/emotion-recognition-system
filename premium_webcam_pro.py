#!/usr/bin/env python3
"""
PREMIUM EMOTION RECOGNITION PRO
Powered by Fusion Logic: MediaPipe Landmarks + Bi-LSTM Deep Learning
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
# Fix for some mediapipe versions
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
from PIL import Image
import sys
import time
from datetime import datetime
from threading import Thread, Lock
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

# ============================================================================
# CONFIGURATION
# ============================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default='2', help='Camera index or URL')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

try:
    CAMERA_INDEX = int(args.camera)
except:
    CAMERA_INDEX = args.camera

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
USE_GPU = args.gpu
SMOOTHING_FACTOR = 0.7  # Temporal blending

# MediaPipe Initialization
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================================
# GLOBAL STATE
# ============================================================================
frame_lock = Lock()
latest_frame = None
should_exit = False

# Model setup
print("📦 Loading AI Engine...")
model = get_model(model_type='full', num_classes=8, pretrained=False, backbone='resnet50', lstm_hidden=256, use_projection=False)
checkpoint_path = r"results\demo_checkpoint\best_model_full_resnet50_20260213_122558.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

device = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
model = model.to(device)
transform = get_transforms('val', img_size=224)
print(f"✅ AI Engine Ready on {device}")

# ============================================================================
# FUSION LOGIC: Geometric Landmarks + Deep Learning
# ============================================================================
def calculate_smile_score(landmarks):
    # Rip corners (61, 291)
    # Upper lip (0), Lower lip (17)
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    upper_lip = landmarks[0]
    lower_lip = landmarks[17]
    
    # Horizontal width
    width = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))
    # Vertical height
    height = np.linalg.norm(np.array([upper_lip.x, upper_lip.y]) - np.array([lower_lip.x, lower_lip.y]))
    
    # Simple ratio
    return width / (height + 1e-6)

def calculate_brow_score(landmarks):
    # Left brow (70), Left eye (159)
    # Right brow (300), Right eye (386)
    left_brow = landmarks[70]
    left_eye = landmarks[159]
    dist = np.linalg.norm(np.array([left_brow.x, left_brow.y]) - np.array([left_eye.x, left_eye.y]))
    return dist

# ============================================================================
# PROCESSING THREADS
# ============================================================================
class CameraThread(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def run(self):
        global latest_frame, should_exit
        while not should_exit:
            ret, frame = self.camera.read()
            if ret:
                with frame_lock:
                    latest_frame = cv2.flip(frame, 1)
            else:
                time.sleep(0.01)
        self.camera.release()

@torch.no_grad()
def get_emotion_probs(face_roi):
    pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return {EMOTION_LABELS[i]: probs[i] for i in range(len(EMOTION_LABELS))}

# ============================================================================
# MAIN UI & LOGIC
# ============================================================================
def main():
    global should_exit, latest_frame
    
    cam_thread = CameraThread()
    cam_thread.start()
    
    cv2.namedWindow('PREMIUM EMOTION PRO', cv2.WINDOW_NORMAL)
    
    last_probs = {label: 0.1 for label in EMOTION_LABELS}
    calibration_frames = 0
    neutral_smile_base = 0.1
    neutral_brow_base = 0.1
    
    fps_start = time.time()
    frames = 0
    fps = 0
    
    print("🚀 System Starting...")
    
    while not should_exit:
        if latest_frame is None:
            continue
            
        with frame_lock:
            img = latest_frame.copy()
            
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        # UI Overlay - Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (w-350, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get Bounding Box from Landmarks
            coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
            x_min, y_min = np.min(coords, axis=0).astype(int)
            x_max, y_max = np.max(coords, axis=0).astype(int)
            
            # Padding
            pad = int((x_max - x_min) * 0.2)
            x1, y1 = max(0, x_min-pad), max(0, y_min-pad)
            x2, y2 = min(w, x_max+pad), min(h, y_max+pad)
            
            face_roi = img[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # 1. Get DL Probs
                raw_probs = get_emotion_probs(face_roi)
                
                # 2. Geometric Feature Extraction
                smile_score = calculate_smile_score(landmarks)
                brow_score = calculate_brow_score(landmarks)
                
                # 3. Calibration (first 30 frames)
                if calibration_frames < 30:
                    neutral_smile_base = (neutral_smile_base * calibration_frames + smile_score) / (calibration_frames + 1)
                    neutral_brow_base = (neutral_brow_base * calibration_frames + brow_score) / (calibration_frames + 1)
                    calibration_frames += 1
                    cv2.putText(img, f"CALIBRATING... {calibration_frames}/30", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 4. FUSION BOOSTING
                # Boost Happy if current smile > base smile
                smile_diff = (smile_score - neutral_smile_base) / neutral_smile_base
                if smile_diff > 0.15:
                    raw_probs['Happy'] *= (1.0 + smile_diff * 2.0)
                
                # Boost Surprise if brows are high
                brow_diff = (brow_score - neutral_brow_base) / neutral_brow_base
                if brow_diff > 0.1:
                    raw_probs['Surprise'] *= (1.0 + brow_diff * 3.0)
                
                # Normalize
                total = sum(raw_probs.values())
                raw_probs = {k: v/total for k, v in raw_probs.items()}
                
                # 5. Temporal Smoothing
                for label in EMOTION_LABELS:
                    last_probs[label] = last_probs[label] * SMOOTHING_FACTOR + raw_probs[label] * (1 - SMOOTHING_FACTOR)
                
                # Detect Final Emotion
                # Suppress Neutral slightly for a "reactive" feel
                display_probs = last_probs.copy()
                display_probs['Neutral'] *= 0.6
                current_emotion = max(display_probs, key=display_probs.get)
                confidence = last_probs[current_emotion]
                
                # DRAWING
                color_map = {
                    'Neutral': (200, 200, 200), 'Happy': (0, 255, 127), 'Sad': (255, 50, 50),
                    'Surprise': (0, 255, 255), 'Fear': (255, 0, 255), 'Disgust': (0, 100, 0),
                    'Anger': (0, 0, 255), 'Contempt': (100, 100, 255)
                }
                color = color_map.get(current_emotion, (255, 255, 255))
                
                # Face Box - Modern Corners
                thickness = 2
                length = 40
                cv2.line(img, (x1, y1), (x1+length, y1), color, thickness*2)
                cv2.line(img, (x1, y1), (x1, y1+length), color, thickness*2)
                cv2.line(img, (x2, y1), (x2-length, y1), color, thickness*2)
                cv2.line(img, (x2, y1), (x2, y1+length), color, thickness*2)
                cv2.line(img, (x1, y2), (x1+length, y2), color, thickness*2)
                cv2.line(img, (x1, y2), (x1, y2+length), color, thickness*2)
                cv2.line(img, (x2, y2), (x2-length, y2), color, thickness*2)
                cv2.line(img, (x2, y2), (x2, y2-length), color, thickness*2)

                # Center Label
                cv2.putText(img, f"{current_emotion.upper()}", (x1, y1-15), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                
                # DASHBOARD BARS
                for i, (label, val) in enumerate(last_probs.items()):
                    y_off = 120 + i*55
                    bar_w = int(val * 200)
                    
                    # Label
                    is_active = (label == current_emotion)
                    text_col = color if is_active else (150, 150, 150)
                    font_w = 2 if is_active else 1
                    cv2.putText(img, label, (w-330, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, font_w)
                    
                    # Track
                    cv2.rectangle(img, (w-200, y_off-15), (w-20, y_off+5), (40, 40, 40), -1)
                    # Bar
                    if bar_w > 0:
                        cv2.rectangle(img, (w-200, y_off-15), (w-200+bar_w, y_off+5), text_col, -1)
                    # %
                    cv2.putText(img, f"{int(val*100)}%", (w-60, y_off-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Header
        cv2.putText(img, "PREMIUM EMOTION ENGINE PRO v2", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.line(img, (30, 60), (450, 60), (0, 255, 255), 2)
        
        # Stats
        frames += 1
        if time.time() - fps_start > 1.0:
            fps = frames / (time.time() - fps_start)
            frames = 0
            fps_start = time.time()
        
        cv2.putText(img, f"FPS: {int(fps)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('PREMIUM EMOTION PRO', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    should_exit = True
    cam_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
