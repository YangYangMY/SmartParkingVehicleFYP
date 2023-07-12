import cv2
from ultralytics import YOLO
import torch
import timeit


# Set to run on GPU device
#torch.cuda.set_device(0)

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

# Create a video capture object

# Open the video file
video_path = "../../../SmartParkingVehicleFYP/ultralytics/VideoFootage/rightCamera1080.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Start the timer
    start = timeit.default_timer()
    # Read a frame from the video
    success, frame = cap.read()



    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame = cv2.resize(annotated_frame, (1280, 720))

        # Stop the timer and calculate the FPS
        end = timeit.default_timer()
        yolo_fps = 1 / (end - start)

        # Display the frame rate on the left top of the screen
        cv2.putText(resized_frame, "FPS: " + "{:.2f}".format(yolo_fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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