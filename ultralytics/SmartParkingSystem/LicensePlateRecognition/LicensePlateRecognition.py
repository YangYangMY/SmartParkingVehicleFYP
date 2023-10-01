import logging
import math
import time

import cvzone
import numpy as np

from SmartParkingSystem.LicensePlateRecognition.sort import Sort
from ultralytics import YOLO
import cv2
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = "LicensePlateRecognition/Tesseract-OCR/tesseract"

model = YOLO("models/yolov8n.pt")
licanese_plate_detector = YOLO("LicensePlateRecognition/models/license_plate_detector.pt")

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

final_ocr_results = []

# Define regular expressions for both formats
pattern1 = re.compile(r'^[A-Z]{3}\d{4}$')
pattern2 = re.compile(r'^[A-Z]{2}\d{4}[A-Z]$')

# --------  Load the video --------
cap = cv2.VideoCapture("LicensePlateRecognition/entryCamera1080.mp4")
# mask = cv2.imread("LicensePlateRecognition/maskLicensePlate.png")
# cap = cv2.VideoCapture("car_video.mp4")

tracker5 = Sort(max_age=120, min_hits=5, iou_threshold=0.3)

# Coordinates of Detection Region
bbox_x1, bbox_y1, bbox_x2, bbox_y2 = 605, 230, 1750, 600
# Coordinates:  [299, 199], [400, 200], [397, 254], [310, 257]

# Coordinates of Detection Region (Region that clear OCR_results)
x1_clear, y1_clear, x2_clear, y2_clear = 1, 329, 560, 600
# Coordinates:  [1, 196], [335, 193], [333, 324], [1, 328]

# Coordinates of Detection Region
detection_region = [(605, 230), (1750,230), (1750,600), (605,600)]
clear_detection_region = [(1, 329), (560, 329), (560, 600), (1, 600)]


stop_all_func = False
car_exist = False
car_plate_exist = False
most_frequent_result = ""
temp = ""

mask = cv2.imread("LicensePlateRecognition/maskImage.png")
car_temp10 = []
ocr_results = {}



def car_detection(frame):
    global ocr_results, final_ocr_results
    imgRegion2 = cv2.bitwise_and(frame, mask)
    results = model(imgRegion2, verbose=False)
    detections5 = np.empty((0, 5))
    clean_result = False

    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" and conf > 0.3:
                currentArray5 = np.array([x1, y1, x2, y2, conf])
                detections5 = np.vstack((detections5, currentArray5))

    resultsTracker5 = tracker5.update(detections5)

    for results in resultsTracker5:
        x1, y1, x2, y2, id = map(int, results)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        clear_region_coordinates = np.array(clear_detection_region, np.int32)
        region_coordinates = np.array(detection_region, np.int32)
        detect_region_result = cv2.pointPolygonTest(region_coordinates, (cx, cy), False)
        clear_region_result = cv2.pointPolygonTest(clear_region_coordinates, (cx, cy), False)


        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                                       offset=10)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if detect_region_result >= 0:
            #Car is inside the detection region
            if id not in car_temp10:
                car_temp10.append(id)

        if clear_region_result >= 0:
            if id in car_temp10:
                car_temp10.remove(id)
                # Stored the most frequent result before cleared
                most_frequent_result = max(ocr_results, key=ocr_results.get, default=None)
                temp = most_frequent_result
                clean_result = True

            # Append the most frequent OCR result after the car leaves the green region but before clear the ocr_results
            if clean_result:
                if temp is not None:
                    final_ocr_results.append(temp)
                    ocr_results = {}
                    clean_result = False


def retrieve_ocr_result():
    global final_ocr_results, ocr_results
    if final_ocr_results:
        first_value = final_ocr_results.pop(0)
        return first_value
    else:
        # Handle the case where the list is empty
        return "Unknown"


def car_plate_video():
    global stop_all_func, temp, car_exist, car_plate_exist
    global most_frequent_result, final_ocr_results
    unknown_stored = False


    try:
        while cap.isOpened():
            ret, frame = cap.read()
            #print("Car Temp Added: ", car_temp)
            if ret and frame is not None:
                # --------  Draw Detection Region Bounding Box --------
                cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 200, 0), 3)

                # --------  Draw Detection Region Bounding Box (For clear) --------
                cv2.rectangle(frame, (x1_clear, y1_clear), (x2_clear, y2_clear), (0, 0, 200), 3)

                # Detect car exist in the green region
                car_detection(frame)

                # Detect license plate
                plateDetections = licanese_plate_detector(frame, stream=True, verbose=False)

                for plateDetection in plateDetections:
                    boxes = plateDetection.boxes

                    for box in boxes:
                        # Detected plate coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = box.conf[0]
                        cls = int(box.cls[0])

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

                            if x1 >= bbox_x1 and y1 >= bbox_y1 and x2 <= bbox_x2 and y2 <= bbox_y2:
                                # print("Plate Detected in region.")
                                cropped = frame[y1:y2, x1:x2]
                                car_plate_exist = True

                                # -------- Preprocess cropped license plate ---------
                                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                                # Resize the image to 2 times
                                resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                                # Blur the image
                                blur = cv2.GaussianBlur(resized, (5, 5), 0)
                                # Thresholding the image
                                rect, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_TOZERO)
                                plate_num = pytesseract.image_to_string(threshold, lang='eng')

                                # --------- Casting ---------
                                # Remove any spaces and convert to uppercase
                                plate_num = re.sub(r'[^a-zA-Z0-9]', '', plate_num)
                                cleaned_result = plate_num.replace(" ", "").upper()

                                # Check if it matches either format
                                if pattern1.match(cleaned_result):
                                    # Format as "ABC 1234"
                                    formatted_result = f"{cleaned_result[:3]} {cleaned_result[3:]}"
                                elif pattern2.match(cleaned_result):
                                    # Format as "AB 1234 C"
                                    formatted_result = f"{cleaned_result[:2]} {cleaned_result[2:6]} {cleaned_result[6]}"
                                else:
                                    if not unknown_stored:
                                        formatted_result = "UNKNOWN"
                                        unknown_stored = True
                                    else:
                                        formatted_result = "UNKNOWN"

                                # Keep track of each identified ocr result
                                if formatted_result in ocr_results:
                                    if unknown_stored is False or formatted_result != "UNKNOWN":
                                        ocr_results[formatted_result] += 1
                                else:
                                    ocr_results[formatted_result] = 1

                                cv2.putText(frame,
                                            "Licence plate number: " + formatted_result,
                                            (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.5,
                                            (0, 0, 255),
                                            4)

                # Play the video
                ResizedframeOutput = cv2.resize(frame, (1060, 640))
                cv2.imshow("OCR Detection", ResizedframeOutput)

                if cv2.waitKey(1) & 0XFF == ord('q'):
                    stop_all_func = True
                    break
            else:
                break

    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")

    # Release the video capture and close
    cap.release()









