#!/usr/bin/env python3
"""
This script will trigger the macOS camera permission dialog
"""
import cv2
import time

print("\n" + "="*70)
print("🔒 TRIGGERING macOS CAMERA PERMISSION REQUEST")
print("="*70)
print("\nWhen the dialog appears:")
print("  ✅ Click 'OK' or 'Allow' to grant camera access to Terminal")
print("\nIf no dialog appears, you need to manually enable camera access:")
print("  1. Open 'System Settings' (System Preferences)")
print("  2. Go to 'Privacy & Security' → 'Camera'")
print("  3. Find 'Terminal' or 'iTerm' in the list")
print("  4. Enable the checkbox next to it")
print("\n" + "="*70 + "\n")

print("Attempting to open camera...")
camera = cv2.VideoCapture(0)

if camera.isOpened():
    print("✅ Camera opened!")
    print("Reading a frame...")
    
    # Give camera time to warm up
    for i in range(10):
        ret, frame = camera.read()
        time.sleep(0.1)
    
    ret, frame = camera.read()
    if ret and frame is not None:
        if frame.max() > 0:
            print(f"✅ SUCCESS! Camera is working! Pixel range: {frame.min()}-{frame.max()}")
            cv2.imwrite('/tmp/permission_test.jpg', frame)
            print("📸 Saved test image to /tmp/permission_test.jpg")
        else:
            print("⚠️ Camera opened but returning black frames")
            print("   Please check System Settings → Privacy & Security → Camera")
    else:
        print("❌ Cannot read from camera")
    
    camera.release()
else:
    print("❌ Cannot open camera")
    print("\n⚠️ ACTION REQUIRED:")
    print("   1. Open System Settings")
    print("   2. Go to Privacy & Security → Camera")
    print("   3. Enable Terminal (or your terminal app)")
    print("   4. Restart Terminal and run this script again")

print("\n" + "="*70)
