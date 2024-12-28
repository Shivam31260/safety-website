from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pathlib

# Adjust the pathlib to work with Windows paths
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Path to your custom YOLOv8 model
path = r"C:\Users\DELL\Desktop\FInal Project\DetectionSystem\best.pt files\test\yolo11n.pt"

# Load the YOLOv8 model
model = YOLO(path)  
cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 5 != 0:  # Process every 5th frame
        continue
    
    # Convert the OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize the PIL image
    pil_image = pil_image.resize((1020, 600))
    
    # Convert the PIL image back to a numpy array for the model
    frame_np = np.array(pil_image)
    
    # Perform inference
    results = model(frame_np)
    
    # The results object is a list; access the first item
    result = results[0]
    
    # Draw bounding boxes on the original frame
    bbox_img = result.plot()  # `plot()` method is used to draw bounding boxes
    
    # Convert the rendered image back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    
    # Display the frame with bounding boxes
    cv2.imshow("FRAME", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

cap.release()
cv2.destroyAllWindows()
