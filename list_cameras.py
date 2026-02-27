import cv2

def list_cameras():
    print("Listing available cameras...")
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    
    if not arr:
        print("No cameras found.")
    else:
        print(f"Cameras found at indices: {arr}")
        print("Check which one is Camo.")

if __name__ == "__main__":
    list_cameras()
