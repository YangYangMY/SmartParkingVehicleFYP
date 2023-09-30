import logging
import time

from SmartParkingSystem.CarDetection.CarDetectController import common_frame_rate
from ultralytics import YOLO
import cv2
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = "LicensePlateRecognition/Tesseract-OCR/tesseract"

model = YOLO("LicensePlateRecognition/models/license_plate_detector.pt")

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


# Coordinates of Detection Region
bbox_x1, bbox_y1, bbox_x2, bbox_y2 = 605, 329, 1016, 546
# Coordinates:  [299, 199], [400, 200], [397, 254], [310, 257]

# Coordinates of Detection Region (Region that clear OCR_results)
x1_clear, y1_clear, x2_clear, y2_clear = 1, 329, 560, 600
# Coordinates:  [1, 196], [335, 193], [333, 324], [1, 328]

stop_all_func = False
most_frequent_result = ""
temp = ""

def retreive_ocr_result():
    global final_ocr_results
    if final_ocr_results:
        first_value = final_ocr_results.pop(0)
        return first_value
    else:
        # Handle the case where the list is empty
        return "Unknown"


def car_plate_video():
    global stop_all_func, temp
    global most_frequent_result
    unknown_stored = False
    clean_result = False
    leave_green_region = False
    ocr_results = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret and frame is not None:
                start_time = time.time()
                # --------  Draw Detection Region Bounding Box --------
                cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 200, 0), 2)

                # --------  Draw Detection Region Bounding Box (For clear) --------
                cv2.rectangle(frame, (x1_clear, y1_clear), (x2_clear, y2_clear), (0, 0, 200), 3)

                # Detect license plate
                plateDetections = model(frame, stream=True, verbose=False)

                for plateDetection in plateDetections:
                    boxes = plateDetection.boxes

                    for box in boxes:
                        # Detected plate coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = box.conf[0]
                        cls = int(box.cls[0])

                        # -------- Crop license plate in Region  ----------
                        if cls == 0 and conf > 0.5:
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
                                # print(cleaned_result)

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

                            # Car leaves the green region
                            if x1 >= x1_clear and y1 >= y1_clear and x2 <= x2_clear and y2 <= y2_clear:
                                # Stored the most frequent result before cleared
                                most_frequent_result = max(ocr_results, key=ocr_results.get, default=None)
                                temp = most_frequent_result
                                clean_result = True

                            # Append the most frequent OCR result after the car leaves the green region but before clear the ocr_results
                            if clean_result:
                                if temp is not None:
                                    final_ocr_results.append(temp)

                                #print("OCR Result cleared.")
                                ocr_results = {}
                                clean_result = False

                           #print("OCR Results:", ocr_results)
                            #print("Most Frequent OCR Result:", most_frequent_result)
                            print("List for pass: ", final_ocr_results)

                # Play the video
                ResizedframeOutput = cv2.resize(frame, (1060, 640))
                cv2.imshow("OCR Detection", ResizedframeOutput)


                # Find the most frequent OCR result
                # most_frequent_result = max(ocr_results, key=ocr_results.get, default="UNKNOWN")
                # print("Most Frequent OCR Result:", most_frequent_result)

                # if len(ocr_results) >= 1:
                #     # Append into a new list
                #     final_ocr_results.append(most_frequent_result)
                #     print(final_ocr_results)


                if cv2.waitKey(1) & 0XFF == ord('q'):
                    stop_all_func = True
                    break


            else:
                break

    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")

    # Release the video capture and close
    cap.release()









