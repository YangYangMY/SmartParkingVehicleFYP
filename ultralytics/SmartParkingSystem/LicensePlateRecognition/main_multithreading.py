from ultralytics import YOLO
import cv2
import pytesseract
import re
from queue import Queue
import threading
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'

model = YOLO("models/license_plate_detector.pt")

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

# Define regular expressions for both formats
pattern1 = re.compile(r'^[A-Z]{3}\d{4}$')
pattern2 = re.compile(r'^[A-Z]{2}\d{4}[A-Z]$')

# --------  Load the video --------
# cap = cv2.VideoCapture("testing.mp4")
cap = cv2.VideoCapture("car_video.mp4")

# Coordinates of Detection Region
bbox_x1, bbox_y1, bbox_x2, bbox_y2 = 605, 329, 1016, 546
# Coordinates:  [299, 199], [400, 200], [397, 254], [310, 257]

# Create a queue to store frames
frame_queue = Queue(maxsize=100)

# Function to read frames and put them into the queue
def read_frames():
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_num += 1
        if not ret:
            break
        frame_queue.put((frame_num, frame))
    cap.release()

# Function to process frames from the queue
def process_frames():
    while not frame_queue.empty():
        frame_num, frame = frame_queue.get()
        print(f"Processing frame {frame_num}")

        plateDetections = model(frame)[0]

        for plateDetection in plateDetections:
            boxes = plateDetection.boxes

            for box in boxes:
                # Detected plate coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[0]
                cls = int(box.cls[0])

                # --------  Draw Detection Region Bounding Box --------
                cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 200, 0), 2)

                # -------- Crop license plate in Region  ----------
                if cls == 0 and conf > 0.3:
                    # Draw License Plate Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"license plate: {conf: .2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    print(x1, y1, x2, y2)
                    if x1 >= bbox_x1 and y1 >= bbox_y1 and x2 <= bbox_x2 and y2 <= bbox_y2:
                        print("Plate Detected in region.")
                        cropped = frame[y1:y2, x1:x2]
                        print("Beep")

                        # -------- Preprocess cropped license plate ---------
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        # Resize the image to 2 times
                        resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        # Blur the image
                        blur = cv2.GaussianBlur(resized, (5, 5), 0)
                        # Thresholding the image
                        rect, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_TOZERO)
                        plate_num = pytesseract.image_to_string(threshold, lang='eng')
                        # print("Beep")

                        # --------- Casting ---------
                        # Remove any spaces and convert to uppercase
                        plate_num = re.sub(r'[^a-zA-Z0-9]', '', plate_num)
                        cleaned_result = plate_num.replace(" ", "").upper()
                        print(cleaned_result)

                        # Check if it matches either format
                        if pattern1.match(cleaned_result):
                            # Format as "ABC 1234"
                            formatted_result = f"{cleaned_result[:3]} {cleaned_result[3:]}"
                        elif pattern2.match(cleaned_result):
                            # Format as "AB 1234 C"
                            formatted_result = f"{cleaned_result[:2]} {cleaned_result[2:6]} {cleaned_result[6]}"
                        else:
                            formatted_result = "Invalid license plate format"

                        print(formatted_result)

                        cv2.putText(frame,
                                    "Licence plate number: " + formatted_result,
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (0, 0, 255),
                                    4)

                # Play the video
                videoOutput = cv2.resize(frame, (1060, 640))
                cv2.imshow("OCR Detection", videoOutput)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the video capture and close
    cap.release()
    cv2.destroyAllWindows()

# Create and start the threads
read_thread = threading.Thread(target=read_frames)
process_thread = threading.Thread(target=process_frames)

read_thread.start()
process_thread.start()

# Wait for the threads to finish
read_thread.join()
process_thread.join()
