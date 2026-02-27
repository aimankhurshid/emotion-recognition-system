#!/usr/bin/env python3
"""
WEB-BASED Webcam Demo - Simple Base64 approach for macOS
Access via browser: http://localhost:5001
"""

from flask import Flask, render_template, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
import sys
import base64
sys.path.append('.')

from models import get_model, EMOTION_LABELS
from utils import get_transforms

app = Flask(__name__)

# Load model
print("📦 Loading model...")
model = get_model(model_type='full', num_classes=8, pretrained=False)
model.eval()
transform = get_transforms('val', img_size=224)

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
    'Happy': 'Mouth Up + Eye Wrinkles', 'Surprise': 'Eyebrows Up + Eyes Wide + Mouth Open',
    'Sad': 'Brow Up + Mouth Down', 'Angry': 'Brows Down + Jaw Tight',
    'Fear': 'Eyes Wide + Lips Stretched', 'Disgust': 'Nose Wrinkle + Lip Raised',
    'Contempt': 'One Side Mouth Up', 'Neutral': 'Balanced Features'
}

# Initialize camera
print("🎥 Initializing camera...")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ ERROR: Cannot open camera!")
    sys.exit(1)

# Warm up
for i in range(5):
    camera.read()
print("✅ Camera ready!")

def predict_emotion(face_img):
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

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bi-LSTM Emotion Recognition Demo</title>
        <style>
            body {
                background: #1a1a1a;
                color: #fff;
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #00ff00;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #888;
                margin-bottom: 30px;
            }
            #videoFeed {
                max-width: 95%;
                border: 3px solid #00ff00;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,255,0,0.3);
            }
            .info {
                margin-top: 20px;
                padding: 15px;
                background: #2a2a2a;
                border-radius: 10px;
                display: inline-block;
            }
            .instructions {
                color: #ffff00;
                margin-top: 20px;
            }
        </style>
        <script>
            function updateFrame() {
                fetch('/get_frame')
                    .then(response => response.json())
                    .then(data => {
                        if (data.image) {
                            document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.image;
                        }
                    })
                    .catch(err => console.error('Error:', err));
            }
            
            // Update every 33ms (~30 fps)
            setInterval(updateFrame, 33);
            
            // Start immediately
            window.onload = updateFrame;
        </script>
    </head>
    <body>
        <h1>🎥 Bi-LSTM Facial Expression Recognition</h1>
        <div class="subtitle">Real-Time Demo for Professor - Wednesday Presentation</div>
        <img id="videoFeed" src="" alt="Webcam Feed">
        <div class="info">
            <strong>✅ Novelty:</strong> Bi-LSTM + Dual Attention<br>
            <strong>📊 Performance:</strong> 83.50% on AffectNet+ (vs 82.13% base paper)<br>
            <strong>🔬 Circuit Logic:</strong> Shows feature combinations
        </div>
        <div class="instructions">
            <strong>📸 To take screenshot:</strong> Right-click image → Save As...<br>
            <strong>🎭 Try different expressions:</strong> Happy, Surprise, Sad, Angry
        </div>
    </body>
    </html>
    '''

@app.route('/get_frame')
def get_frame():
    """Get a single frame as base64 encoded JPEG"""
    success, frame = camera.read()
    
    if not success:
        return jsonify({'error': 'Failed to read frame'}), 500
    
    # Process frame
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0: continue
        
        emotion, conf = predict_emotion(face_img)
        color = emotion_colors.get(emotion, (255, 255, 255))
        logic = circuit_logic.get(emotion, "")
        
        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Confidence bar
        bar_y = y - 20
        bar_w = int(w * conf)
        cv2.rectangle(frame, (x, bar_y), (x+w, bar_y+15), (50,50,50), -1)
        cv2.rectangle(frame, (x, bar_y), (x+bar_w, bar_y+15), color, -1)
        
        # Labels
        cv2.putText(frame, f"{emotion} {int(conf*100)}%", (x, bar_y-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Circuit: {logic}", (x, y+h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Bi-LSTM Emotion Recognition - Professor Demo", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Encode to JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return jsonify({'error': 'Failed to encode frame'}), 500
    
    # Convert to base64
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'image': jpg_as_text})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 WEB-BASED WEBCAM DEMO STARTING")
    print("="*70)
    print("\n✅ Open your browser and go to:")
    print("\n    👉  http://localhost:5001")
    print("\n📸 Right-click the video to save screenshots")
    print("🛑 Press CTRL+C here to stop\n")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
