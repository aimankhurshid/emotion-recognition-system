"""
Real-time emotion recognition from webcam
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, load_checkpoint, EMOTION_LABELS
from utils import get_transforms


class EmotionRecognizer:
    """Real-time emotion recognition system"""
    
    def __init__(self, model_path, model_type='full', backbone='efficientnet_b4', 
                 num_classes=8, img_size=224, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.transform = get_transforms('val', img_size=img_size)
        
        print(f"Loading model from {model_path}...")
        self.model = get_model(
            model_type=model_type,
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )
        self.model = self.model.to(self.device)
        load_checkpoint(self.model, None, model_path, device=self.device)
        self.model.eval()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_colors = {
            'Neutral': (200, 200, 200),
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Surprise': (255, 255, 0),
            'Fear': (128, 0, 128),
            'Disgust': (0, 128, 0),
            'Anger': (0, 0, 255),
            'Contempt': (128, 128, 0)
        }
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image"""
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
            
            emotion_id = predicted.item()
            emotion_label = EMOTION_LABELS[emotion_id]
            confidence_score = confidence.item()
            
            return emotion_label, confidence_score
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0
    
    def draw_results(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion label on frame"""
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        label = f"{emotion}: {confidence*100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    def run(self, camera_id=0, output_video=None):
        """Run real-time emotion recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_time = 0
        frame_count = 0
        
        video_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        print("\n" + "="*60)
        print("Real-time Emotion Recognition - Press 'q' to quit")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            
            for (x, y, w, h) in faces:
                padding = int(0.1 * max(w, h))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_image = frame[y1:y2, x1:x2]
                
                if face_image.size > 0:
                    emotion, confidence = self.predict_emotion(face_image)
                    self.draw_results(frame, x, y, w, h, emotion, confidence)
            
            current_time = time.time()
            fps = 1 / (current_time - fps_time) if fps_time > 0 else 0
            fps_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            
            if video_writer:
                video_writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"Video saved to {output_video}")
        
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main(args):
    recognizer = EmotionRecognizer(
        model_path=args.model_path,
        model_type=args.model_type,
        backbone=args.backbone,
        num_classes=args.num_classes,
        img_size=args.img_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    recognizer.run(camera_id=args.camera_id, output_video=args.output_video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition from Webcam')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Path to save output video (optional)')
    
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
    
    args = parser.parse_args()
    
    main(args)
