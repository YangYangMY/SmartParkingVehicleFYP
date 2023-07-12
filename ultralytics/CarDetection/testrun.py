import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../Models/yolov8x.pt')
model.classes = [0, 2]

results = model.predict(source="rightCamera1080.mp4", save=True, imgsz=640, conf=0.25, device=0)


print(results)




