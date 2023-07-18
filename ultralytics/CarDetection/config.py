import math
import cv2
from ultralytics import YOLO
import torch
import timeit
import cvzone
from CarDetection.sort import *
from datetime import datetime

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

#Store Car Dictionary
copy_car_dict = {"carPlate": "unknown" ,"bbox": "unknown", "entryTime": "unknown", "exitTime": "unknown", "status": "moving", "currentLocation": "unknown"}
car_dict = {}

#Temporary Car Dictionary
car_temp = {} #Right Cam Bottom Exit Line
car_temp2 = {} #Middle Cam Bottom Entry Line
car_temp3 = {} #Right Cam Middle Exit Line
car_temp4 = {} #Middle Cam Middle Entry Line
car_temp5 = {} #Middle Cam Bottom Exit Line
car_temp6 = {} #Left Cam Bottom Entry Line
car_temp7 = {} #Middle Cam Middle Exit Line
car_temp8 = {} #Left Cam Middle Entry Line


# Open the video file 1
video_path1 = "../VideoFootage/rightCamMiddleCar.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("maskRightCamera.png")

# Open the video file 2
video_path2 = "../VideoFootage/midCamMiddleCar.mp4"
cap2 = cv2.VideoCapture(video_path2)
mask2 = cv2.imread("maskMiddleCamera.png")

# Open the video file 3
video_path3 = "../VideoFootage/leftCamMiddleCar.mp4"
cap3 = cv2.VideoCapture(video_path3)
mask3 = cv2.imread("maskLeftCamera3.png")



# Load the YOLO v8 pre-trained model
modelm = YOLO('../Models/yolov8m.pt')
modelx = YOLO('../Models/yolov8x.pt')

# Tracking store id for each object
tracker = Sort(max_age=120, min_hits=5, iou_threshold=0.3) #Store for right Camera
tracker2 = Sort(max_age=120, min_hits=5, iou_threshold=0.3) #Store for middle Camera
tracker3 = Sort(max_age=120, min_hits=5, iou_threshold=0.3) #Store for left Camera

# Store the number of cars that crossed the line
totalCount = []

# Store the coordinates for right camera
RightCamEntryLine = [1020, 175, 1157, 158]
RightCamMiddleLine = [700, 210, 730, 250]
RightCamBottomLine = [930, 430, 1180, 550]

# Store the coordinates for middle camera
MiddleCamMiddleEntryLine = [1780, 170, 1810, 210]
MiddleCamMiddleExitLine = [840, 178, 860, 230]

MiddleCamBottomEntryLine = [1830, 510, 1800, 880]
MiddleCamBottomExitLine = [690, 510, 590, 885]

# Store the coordinates for left camera
LeftCamMiddleEntryLine = [1730, 390, 1670, 400]
LeftCamMiddleExitLine = [1050, 270, 1000, 290]

LeftCamBottomEntryLine = [1300, 620, 1200, 880]
LeftCamBottomExitLine = [300, 400, 550, 340]

#Store each Park Lot Information
#A1- A20, B1- B20, C1- C15, D1- D19
ParkingLot = {}
ParkingLot_empty = {"carId": "unknown","firstDetected": "unknown", "duration": "unknown", "parked": "no"}
for i in range(20):
    i += 1
    ParkingLot['A'+ str(i)] = ParkingLot_empty.copy()
    ParkingLot['B'+ str(i)] = ParkingLot_empty.copy()

for i in range(15):
    i += 1
    ParkingLot['C'+ str(i)] = ParkingLot_empty.copy()

for i in range(19):
    i += 1
    ParkingLot['D'+ str(i)] = ParkingLot_empty.copy()

#Store Location for each Parking Lot
ParkingLot_A1 = [(837,205),(795,216),(746,186),(787,172)]
ParkingLot_A2 = [(769,209),(725,217),(690,186),(727,180)]
ParkingLot_A3 = [(697,230),(651,233),(610,215),(659,193)]