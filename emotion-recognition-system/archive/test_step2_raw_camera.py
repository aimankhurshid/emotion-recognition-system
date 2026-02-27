#!/usr/bin/env python3
"""
STEP 2: Test raw camera display (NO processing, NO face detection, NO model)
"""
import cv2
import sys

print("STEP 2: Testing RAW camera display...")

# Open camera
print("Opening camera...")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ ERROR: Cannot open camera!")
    sys.exit(1)

print("✅ Camera opened!")

# Read one frame first to verify
ret, test_frame = camera.read()
if not ret or test_frame is None:
    print("❌ ERROR: Cannot read frame from camera!")
    camera.release()
    sys.exit(1)

print(f"✅ Frame captured! Shape: {test_frame.shape}")
print(f"   Min pixel value: {test_frame.min()}, Max: {test_frame.max()}")

# Create window
print("Creating window...")
cv2.namedWindow('Raw Camera Test', cv2.WINDOW_NORMAL)

print("\n🎥 Showing camera feed...")
print("   Press 'q' to quit")
print("   You should see YOUR FACE now!\n")

frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret:
        print(f"⚠️ Failed to read frame at count {frame_count}")
        break
    
    frame_count += 1
    
    # Just flip and show - NO PROCESSING
    frame = cv2.flip(frame, 1)
    
    # Add frame counter to verify it's updating
    cv2.putText(frame, f"Frame: {frame_count}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Raw Camera Test', frame)
    
    # Check for 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n✅ Processed {frame_count} frames")
camera.release()
cv2.destroyAllWindows()
