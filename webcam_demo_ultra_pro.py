#!/usr/bin/env python3
"""
ULTRA-PRO EMOTION RECOGNITION SYSTEM
Highly Sensitive | Fully Optimized | Modern UI
"""

import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
import time
from datetime import datetime
from threading import Thread, Lock
import urllib.request

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
parser.add_argument('--sensitivity', type=float, default=1.5, help='Emotion boost factor')
args = parser.parse_args()

try:
    CAMERA_INDEX = int(args.camera)
except:
    CAMERA_INDEX = args.camera

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SENSITIVITY = args.sensitivity
USE_GPU = args.gpu or torch.cuda.is_available()

# DNN Face Detector Files
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTO_FILE = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("✅ Downloaded!")

download_file(PROTO_URL, PROTO_FILE)
download_file(MODEL_URL, MODEL_FILE)

# Load DNN Detector
face_net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
if USE_GPU:
    face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# ============================================================================
# AI MODEL LOADING
# ============================================================================
print("📦 Initializing AI Engine...")
model = get_model(model_type='full', num_classes=8, pretrained=False, backbone='resnet50', lstm_hidden=256, use_projection=False)
checkpoint_path = r"results\demo_checkpoint\best_model_full_resnet50_20260213_122558.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

device = torch.device('cuda' if USE_GPU else 'cpu')
model = model.to(device)
transform = get_transforms('val', img_size=224)
print(f"✅ AI Ready on {device}")

# ============================================================================
# STATE & THREADING
# ============================================================================
frame_lock = Lock()
latest_frame = None
should_exit = False

class CameraThread(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def run(self):
        global latest_frame, should_exit
        while not should_exit:
            ret, frame = self.cap.read()
            if ret:
                with frame_lock:
                    latest_frame = cv2.flip(frame, 1)
            else:
                time.sleep(0.01)
        self.cap.release()

# ============================================================================
# PROCESSING HUB
# ============================================================================
@torch.no_grad()
def predict(face_roi):
    img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return {EMOTION_LABELS[i]: probs[i] for i in range(len(EMOTION_LABELS))}

def main():
    global should_exit, latest_frame, SENSITIVITY
    
    cam = CameraThread()
    cam.start()
    
    cv2.namedWindow('ULTRA-PRO EMOTION ENGINE', cv2.WINDOW_NORMAL)
    
    # Smoothing & Baseline
    smooth_probs = {em: 0.1 for em in EMOTION_LABELS}
    fps_start = time.time()
    frames = 0
    fps = 0
    
    print("\n🚀 System Online. Ready for detection.")
    
    while not should_exit:
        if latest_frame is None: continue
        
        with frame_lock:
            frame = latest_frame.copy()
            
        h, w = frame.shape[:2]
        
        # 1. DNN Face Detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-320, 0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Expand slightly for context
                pad = int((x2-x1) * 0.15)
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0: continue
                
                # 2. AI Inference
                raw_probs = predict(face_roi)
                
                # 3. SENSITIVITY CALIBRATION
                # Bias correction: Lower Neutral, Boost the rest
                raw_probs['Neutral'] *= 0.4
                for em in ['Happy', 'Surprise', 'Anger', 'Sad']:
                    raw_probs[em] *= SENSITIVITY
                
                # Normalize
                total = sum(raw_probs.values())
                raw_probs = {k: v/total for k, v in raw_probs.items()}
                
                # 4. Temporal Smoothing
                alpha = 0.6
                for em in EMOTION_LABELS:
                    smooth_probs[em] = smooth_probs[em] * alpha + raw_probs[em] * (1-alpha)
                
                # 5. UI Drawing
                current_em = max(smooth_probs, key=smooth_probs.get)
                conf = smooth_probs[current_em]
                
                colors = {
                    'Neutral': (150, 150, 150), 'Happy': (0, 255, 127), 'Sad': (255, 50, 50),
                    'Surprise': (0, 255, 255), 'Fear': (255, 0, 255), 'Disgust': (0, 100, 0),
                    'Anger': (0, 0, 255), 'Contempt': (100, 100, 255)
                }
                color = colors.get(current_em, (255,255,255))
                
                # Modern Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, -1)
                cv2.putText(frame, f"{current_em.upper()} {int(conf*100)}%", (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                
                # Sidebar Dashboard
                for j, (em, val) in enumerate(smooth_probs.items()):
                    y_pos = 100 + j*55
                    bar_w = int(val * 180)
                    active = (em == current_em)
                    
                    tc = color if active else (180, 180, 180)
                    cv2.putText(frame, em, (w-300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2 if active else 1)
                    
                    # Track
                    cv2.rectangle(frame, (w-190, y_pos-12), (w-10, y_pos+3), (50, 50, 50), -1)
                    # Bar
                    if bar_w > 0:
                        cv2.rectangle(frame, (w-190, y_pos-12), (w-190+bar_w, y_pos+3), tc, -1)

        # Header
        cv2.putText(frame, "ULTRA-PRO EMOTION ENGINE v3.0", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.line(frame, (30, 60), (480, 60), (0, 255, 255), 2)
        
        # FPS
        frames += 1
        if time.time() - fps_start > 1.0:
            fps = frames / (time.time() - fps_start)
            frames = 0
            fps_start = time.time()
        cv2.putText(frame, f"FPS: {int(fps)} | SENSITIVITY: {SENSITIVITY}x", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('ULTRA-PRO EMOTION ENGINE', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('+'): SENSITIVITY += 0.1
        elif key == ord('-'): SENSITIVITY = max(1.0, SENSITIVITY - 0.1)
        
    should_exit = True
    cam.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
