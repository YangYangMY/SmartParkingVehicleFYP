import math

import cv2
from ultralytics import YOLO
import torch
import timeit
import cvzone
from CarDetection.sort import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if torch.cuda.is_available():
    print("Running on GPU")
    # Set to run on GPU device
    torch.cuda.set_device(0)
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

# Open the video file 1
video_path1 = "../VideoFootage/rightCamera1080.mp4"
cap1 = cv2.VideoCapture(video_path1)
#cap1 = cv2.VideoCapture(0)

# Open the video file 2
video_path2 = "../VideoFootage/middleCamera1080.mp4"
cap2 = cv2.VideoCapture(video_path2)
#cap2 = cv2.VideoCapture(0)

# Check if video files were opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video files")
    exit()

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [1020, 175, 1157, 158]
totalCount = []

# Loop through the video frames
while cap1.isOpened() and cap2.isOpened():
    # Start the timer
    start = timeit.default_timer()

    # Read a frame from the video
    success1, frame1 = cap1.read()

    if success1:
        # Run YOLOv8 inference on the frame
        results = model(frame1, stream=True)

        detections = np.empty((0,5))

        # Store the ids of the cars that crossed the line
        crossed_ids = []

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
                    #cvzone.putTextRect(frame1, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                   #                scale=0.9, thickness=1, offset=5)
                   # cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=5)
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(frame1,(limits[0], limits[1]), (limits[2], limits[3]),(0,0,255), 5)


        for results in resultsTracker:
            x1,y1,x2,y2,id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(results)
            w, h = x2 - x1, y2 - y1
          #  cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
          #  cvzone.putTextRect(frame1, f' {int(id)}', (max(0, x1), max(35, y1)),
          #                     scale=2, thickness=3, offset=10)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(frame1, (cx,cy), 5, (255,0,255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] > cy > limits[3]:
                if totalCount.count(id) == 0 and id not in crossed_ids:
                    totalCount.append(id)
                    crossed_ids.append(id)
                    cv2.line(frame1, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Draw corner rectangle for specific id
        if len(totalCount) > 0:
            specific_id = totalCount[-1]  # Get the last appended id
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                if id == specific_id:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
                    cvzone.putTextRect(frame1, f' {int(id)}', (max(0, x1), max(35, y1)),
                                        scale=2, thickness=3, offset=10)

        # Read a frame from the video
        success2, frame2 = cap2.read()

        if success2:
            # Run YOLOv8 inference on the frame
            results = model(frame2, stream=True)

            detections = np.empty((0, 5))

            # Store the ids of the cars that crossed the line
            crossed_ids = []

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
                        #cvzone.putTextRect(frame2, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                        #                scale=2, thickness=2, offset=5)
                        #cvzone.cornerRect(frame2, (x1, y1, w, h), l=9, rt=5)
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))



        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame1 = cv2.resize(frame1, (800, 600))
        resized_frame2 = cv2.resize(frame2, (800, 600))
        #resized_frame1 = cv2.resize(frame1, (1280, 720))
        #resized_frame2 = cv2.resize(frame2, (1280, 720))

        # Display the frame rate on the left top of the screen
        cv2.putText(resized_frame1, f' Count: {len(totalCount)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(resized_frame2, f' Count: {len(totalCount)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Stop the timer and calculate the FPS
        end = timeit.default_timer()
        yolo_fps = 1 / (end - start)

        # Display the frame rate on the left top of the screen
        cv2.putText(resized_frame1, "FPS: " + "{:.2f}".format(yolo_fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.putText(resized_frame2, "FPS: " + "{:.2f}".format(yolo_fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Combine the frames side by side
        combined_frame = cv2.hconcat([resized_frame2, resized_frame1])

        # Display the combined frame
        cv2.imshow("Combined Videos", combined_frame)

        # Display the annotated frame
       # cv2.imshow("YOLOv8 Car Detection 1", resized_frame1)
       # cv2.imshow("YOLOv8 Car Detection 2", resized_frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap1.release()
cap2.release()
cv2.destroyAllWindows()