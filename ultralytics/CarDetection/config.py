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
copy_car_dict = {"carPlate": "unknown" ,"bbox": "unknown", "entryTime": "unknown",
                 "exitTime": "unknown", "status": "moving", "currentLocation": "unknown",
                 "parkingLot": "unknown", "parkingDetected": "unknown", "duration": "unknown"}
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
video_path1 = "../VideoFootage/rightCam3.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("maskRightCamera1.png")

# Open the video file 2
video_path2 = "../VideoFootage/midCam3.mp4"
cap2 = cv2.VideoCapture(video_path2)
mask2 = cv2.imread("maskMiddleCamera.png")

# Open the video file 3
video_path3 = "../VideoFootage/leftCam3.mp4"
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
ParkingLot_empty = {"carId": "unknown", "parked": "no"}
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

#Store Location for each Parking Lot for Right Camera
ParkingLot_A1 = [(1814,354),(1693,337),(1715,307),(1837,323)]

ParkingLot_B1 = [(1316,261),(1292,281),(1214,232),(1248,224)]
ParkingLot_B2 = [(1264,283),(1243,297),(1142,255),(1198,241)]
ParkingLot_B3 = [(1203,304),(1166,316),(1083,276),(1105,250)]
ParkingLot_B4 = [(1164,339),(1118,353),(1030,307),(1059,286)]
ParkingLot_B5 = [(1080,346),(1041,379),(943,317),(974,286)]
ParkingLot_B6 = [(988,385),(928,401),(817,360),(870,304)]
ParkingLot_B7 = [(895,426),(821,444),(707,389),(761,313)]

ParkingLot_D1 = [(837,205),(795,216),(746,186),(787,172)]
ParkingLot_D2 = [(769,209),(725,217),(690,186),(727,180)]

#Store Location for each Parking Lot for Middle Camera
ParkingLot_B8 = [(1755,464),(1658,465),(1611,342),(1715,357)]

ParkingLot_C1 = [(1827,280),(1774,275),(1742,230),(1816,233)]
ParkingLot_C2 = [(1705,273),(1635,274),(1614,231),(1675,225)]
ParkingLot_C3 = [(1564,274),(1502,274),(1484,235),(1533,227)]
ParkingLot_C4 = [(1401,273),(1327,273),(1312,237),(1375,227)]
ParkingLot_C5 = [(1229,271),(1167,271),(1150,231),(1216,227)]
ParkingLot_C6 = [(1091,271),(1008,269),(1012,230),(1076,227)]
ParkingLot_C7 = [(934,284),(864,284),(861,239),(937,233)]
ParkingLot_C8 = [(778,276),(704,276),(713,235),(776,235)]
ParkingLot_C9 = [(631,276),(565,276),(580,235),(632,235)]
ParkingLot_C10 = [(458,276),(402,276),(410,230),(472,230)]
ParkingLot_C11 = [(297,276),(232,276),(241,234),(306,234)]

ParkingLot_D3 = [(1757,160),(1702,161),(1682,127),(1738,127)]
ParkingLot_D4 = [(1655,164),(1598,164),(1571,127),(1630,127)]
ParkingLot_D5 = [(1545,161),(1498,166),(1482,127),(1537,127)]
ParkingLot_D6 = [(1444,164),(1392,159),(1380,115),(1427,117)]
ParkingLot_D7 = [(1363,157),(1309,154),(1300,115),(1345,117)]
ParkingLot_D8 = [(1263,159),(1203,156),(1205,120),(1249,123)]
ParkingLot_D9 = [(1152,161),(1101,158),(1092,121),(1143,121)]
ParkingLot_D10 = [(1045,162),(1001,164),(991,121),(1037,121)]
ParkingLot_D11 = [(933,161),(880,158),(883,108),(941,110)]
ParkingLot_D12 = [(843,157),(777,160),(785,107),(848,109)]
ParkingLot_D13 = [(726,152),(665,152),(675,107),(727,111)]
ParkingLot_D14 = [(604,160),(557,154),(564,110),(607,113)]
ParkingLot_D15 = [(519,165),(477,164),(480,110),(528,113)]
ParkingLot_D16 = [(418,165),(376,164),(380,110),(427,113)]