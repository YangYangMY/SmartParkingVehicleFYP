import math

import cv2
from ultralytics import YOLO
import torch
import timeit
import cvzone
from CarDetection.sort import *

# Set to run on GPU device
torch.cuda.set_device(0)

if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

# Load the YOLO v8 pre-trained model
model = YOLO('../Models/yolov8m.pt')


# Define the list of classes that we want to detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Open the video file
video_path = "../../../SmartParkingVehicleFYP/ultralytics/VideoFootage/rightCamera1080.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [1020, 175, 1157, 158]
totalCount = []

# Loop through the video frames
while cap.isOpened():
    # Start the timer
    start = timeit.default_timer()
    # Read a frame from the video
    success, frame = cap.read()



    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        detections = np.empty((0,5))

        # Visualize the results on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                currentClass = classNames[cls]

                if currentClass == "car" and conf > 0.3:
                    #cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                   #                scale=0.9, thickness=1, offset=5)
                   # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(frame,(limits[0], limits[1]), (limits[2], limits[3]),(0,0,255), 5)


        for results in resultsTracker:
            x1,y1,x2,y2,id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(results)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
            cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] > cy > limits[3]:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame = cv2.resize(frame, (1280, 720))

        # Display the frame rate on the left top of the screen
        cv2.putText(resized_frame, f' Count: {len(totalCount)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Stop the timer and calculate the FPS
        end = timeit.default_timer()
        yolo_fps = 1 / (end - start)

        # Display the frame rate on the left top of the screen
        cv2.putText(resized_frame, "FPS: " + "{:.2f}".format(yolo_fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Car Detection", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()