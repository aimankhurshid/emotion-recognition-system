#!/usr/bin/env python3
"""Simple camera test to verify OpenCV can access the webcam"""

import cv2
import sys

print("🎥 Testing camera access...")

# Try to open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Cannot open camera!")
    sys.exit(1)

print("✅ Camera opened successfully!")
print("📸 Trying to read a frame...")

# Try to read a frame
ret, frame = camera.read()

if not ret or frame is None:
    print("❌ Cannot read frame from camera!")
    camera.release()
    sys.exit(1)

print(f"✅ Frame captured successfully! Shape: {frame.shape}")
print(f"   Width: {frame.shape[1]}, Height: {frame.shape[0]}")

# Try to save the frame
cv2.imwrite('/tmp/test_frame.jpg', frame)
print("✅ Test frame saved to /tmp/test_frame.jpg")

camera.release()
print("\n🎉 Camera test PASSED! Your webcam is working correctly.")
