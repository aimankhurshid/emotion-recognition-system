#!/usr/bin/env python3
"""
STEP 3: Deep camera diagnostics - check permissions and settings
"""
import cv2
import subprocess
import sys

print("STEP 3: Camera Deep Diagnostics")
print("="*70)

# Test 1: Check camera permissions via system
print("\n1. Checking Terminal camera permissions...")
result = subprocess.run(['tccutil', 'reset', 'Camera'], capture_output=True, text=True)
print(f"   Reset camera permissions (you may need to re-grant)")

# Test 2: Try multiple camera backends
print("\n2. Testing different camera backends...")

backends = [
    (cv2.CAP_ANY, "CAP_ANY (auto)"),
    (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION (macOS native)"),
]

for backend_id, backend_name in backends:
    print(f"\n   Testing: {backend_name}")
    try:
        cam = cv2.VideoCapture(0, backend_id)
        if not cam.isOpened():
            print(f"   ❌ Failed to open with {backend_name}")
            continue
        
        ret, frame = cam.read()
        if not ret or frame is None:
            print(f"   ❌ Cannot read frame with {backend_name}")
            cam.release()
            continue
        
        print(f"   ✅ Opened! Frame shape: {frame.shape}")
        print(f"      Pixel values - Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.2f}")
        
        # Check if frame is all black
        if frame.max() == 0:
            print(f"      ⚠️  WARNING: ALL BLACK PIXELS!")
        else:
            print(f"      ✅ Frame has actual data!")
            
            # Save a test frame
            cv2.imwrite(f'/tmp/test_frame_{backend_name.replace(" ", "_")}.jpg', frame)
            print(f"      📸 Saved test frame to /tmp/test_frame_{backend_name.replace(' ', '_')}.jpg")
        
        cam.release()
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test 3: Check camera properties
print("\n3. Detailed camera properties check...")
camera = cv2.VideoCapture(0)
if camera.isOpened():
    props = {
        'FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
        'FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
        'CONTRAST': cv2.CAP_PROP_CONTRAST,
        'SATURATION': cv2.CAP_PROP_SATURATION,
        'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
        'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
    }
    
    for name, prop in props.items():
        value = camera.get(prop)
        print(f"   {name}: {value}")
    
    # Try adjusting exposure
    print("\n4. Trying to fix exposure settings...")
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto-exposure
    camera.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust exposure
    
    # Wait a bit and try again
    import time
    print("   Waiting 2 seconds for camera to adjust...")
    time.sleep(2)
    
    # Read multiple frames to give camera time to warm up
    print("   Reading 10 frames to warm up camera...")
    for i in range(10):
        ret, frame = camera.read()
        if ret and frame is not None:
            print(f"   Frame {i+1}: Min={frame.min()}, Max={frame.max()}, Mean={frame.mean():.2f}")
    
    # Try one more time
    ret, final_frame = camera.read()
    if ret and final_frame is not None:
        print(f"\n   Final test frame: Min={final_frame.min()}, Max={final_frame.max()}")
        if final_frame.max() > 0:
            cv2.imwrite('/tmp/test_frame_fixed.jpg', final_frame)
            print("   ✅ SUCCESS! Saved working frame to /tmp/test_frame_fixed.jpg")
        else:
            print("   ❌ Still getting black frames")
    
    camera.release()

print("\n" + "="*70)
print("Diagnostics complete!")
