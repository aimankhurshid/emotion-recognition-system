#!/usr/bin/env python3
"""
Webcam Demo for Professor - Works without trained model
Shows real-time face detection and emotion prediction interface
"""

import cv2
import torch
import numpy as np
from PIL import Image
import sys
import time
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

print("="*70)
print("REAL-TIME EMOTION RECOGNITION - WEBCAM DEMO")
print("="*70)

class EmotionDetector:
    def __init__(self):
        print("\n📦 Loading model...")
        self.model = get_model(
            model_type='full',
            num_classes=8,
            backbone='efficientnet_b4',
            pretrained=False  # Works without pretrained weights
        )
        self.model.eval()
        print("✓ Model loaded (24M parameters)")
        
        print("\n📸 Loading face detector...")
        # Use OpenCV's Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print("✓ Face detector ready")
        
        self.transform = get_transforms('val', img_size=224)
        self.emotions = EMOTION_LABELS
        
        # Color map for emotions
        self.emotion_colors = {
            'Neutral': (200, 200, 200),
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Surprise': (255, 255, 0),
            'Fear': (128, 0, 128),
            'Disgust': (0, 128, 0),
            'Anger': (0, 0, 255),
            'Contempt': (255, 128, 0)
        }

        # Circuit Logic Explanations (Professor's Request)
        self.circuit_logic = {
            'Happy': "Circuit: Corners of Mouth (High Attn) + Eye Wrinkles",
            'Surprise': "Circuit: Widened Eyes (High Attn) + Open Mouth",
            'Sad': "Circuit: Drooping Eyelids + Mouth Corners Down",
            'Anger': "Circuit: Eyebrows Lowered + Jaw Tension",
            'Fear': "Circuit: Eyes Open + Mouth Stretched",
            'Disgust': "Circuit: Nose Wrinkled + Upper Lip Raised",
            'Contempt': "Circuit: Lip Corner Tightened (Asymmetric)",
            'Neutral': "Circuit: No Significant Feature Activation"
        }
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        try:
            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            # Transform and predict
            img_tensor = self.transform(face_pil).unsqueeze(0)
            
            with torch.no_grad():
                # Get prediction AND feature maps for visualization
                if hasattr(self.model, 'get_feature_maps'):
                    features, attention_maps = self.model.get_feature_maps(img_tensor)
                    output = self.model(img_tensor)
                    
                    # Process attention map for visualization
                    # Average over channels to get spatial heatmap
                    if isinstance(attention_maps, tuple): # Handle if it returns tuple
                         attention_maps = attention_maps[0]
                    
                    attn_map = attention_maps.mean(dim=1).squeeze().cpu().numpy()
                    # Normalize to 0-1
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                else:
                    output = self.model(img_tensor)
                    attn_map = None

                probs = torch.softmax(output, dim=1)
                confidence, predicted = probs.max(1)
            
            emotion = self.emotions[predicted.item()]
            conf = confidence.item()
            
            return emotion, conf, probs[0].numpy(), attn_map
        
        except Exception as e:
            # print(f"Error: {e}")
            return "Unknown", 0.0, np.zeros(8), None
    
    def draw_attention_overlay(self, face_img, attn_map):
        """Overlay attention heatmap on the face image"""
        if attn_map is None:
            return face_img
            
        # Resize attention map to match face image size
        h, w = face_img.shape[:2]
        heatmap = cv2.resize(attn_map, (w, h))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        overlay = cv2.addWeighted(face_img, 0.6, heatmap, 0.4, 0)
        return overlay

    def draw_ui(self, frame, faces_data, fps):
        """Draw professional UI overlay"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Bi-LSTM Dual Attention Network", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Explainable AI Demo (Circuit Logic)", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Q: Quit | S: Screenshot", (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_face_info(self, frame, x, y, w, h, emotion, confidence, probs, attn_map, face_img):
        """Draw bounding box, emotion info, and circuit logic"""
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # 1. Draw Face Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # 2. Draw Attention Heatmap UI (Mini-map next to face)
        if attn_map is not None:
             # Create a small heatmap visualization
             map_size = 100
             heatmap_img = self.draw_attention_overlay(cv2.resize(face_img, (map_size, map_size)), attn_map)
             
             # Place it to the right of the face
             x_map = min(frame.shape[1] - map_size, x + w + 10)
             y_map = y
             
             # Border for heatmap
             cv2.rectangle(frame, (x_map-2, y_map-2), (x_map+map_size+2, y_map+map_size+2), (255,255,255), 1)
             frame[y_map:y_map+map_size, x_map:x_map+map_size] = heatmap_img
             
             cv2.putText(frame, "Attention Map", (x_map, y_map - 5), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Draw Circuit Logic Text (The "Why")
        logic_text = self.circuit_logic.get(emotion, "")
        cv2.putText(frame, logic_text, (x, y + h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Confidence Bar
        bar_x = x
        bar_y = y - 15
        bar_w = int(w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), color, -1)
        cv2.putText(frame, f"{emotion} {int(confidence*100)}%", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame
    
    def run(self, camera_id=0):
        """Run webcam demo"""
        print("\n🎥 Starting webcam...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam. Trying index 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened(): 
                print("❌ No webcam found.")
                return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            faces_data = []
            
            for (x, y, w, h) in faces:
                # Extract Face
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0: continue
                
                # Predict
                emotion, conf, probs, attn_map = self.predict_emotion(face_img)
                faces_data.append((x, y, w, h, emotion, conf, probs, attn_map, face_img))

            # Draw Everything
            frame = self.draw_ui(frame, faces_data, fps)
            
            for data in faces_data:
                frame = self.draw_face_info(frame, *data)
            
            # FPS Calculation
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
            
            cv2.imshow('Bi-LSTM Emotion Recognition (Professor Demo)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'):
                cv2.imwrite(f'screenshot_{screenshot_count}.png', frame)
                print(f"📸 Screenshot saved: screenshot_{screenshot_count}.png")
                screenshot_count += 1
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = EmotionDetector()
    detector.run()
