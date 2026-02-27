#!/usr/bin/env python3
"""
STEP 1: Test if OpenCV can display ANYTHING at all
"""
import cv2
import numpy as np

print("STEP 1: Testing basic OpenCV display...")

# Create a simple colored image (not from camera)
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
test_img[:, :] = (0, 255, 0)  # Green

# Add text
cv2.putText(test_img, "TEST - If you see this, OpenCV display works!", 
           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

print("Creating test window...")
cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
cv2.imshow('Test Window', test_img)

print("✅ Window created and image shown")
print("⏳ Waiting 5 seconds... (do you see a GREEN window with text?)")

# Wait 5 seconds
cv2.waitKey(5000)

print("Closing window...")
cv2.destroyAllWindows()
print("✅ Test complete")
