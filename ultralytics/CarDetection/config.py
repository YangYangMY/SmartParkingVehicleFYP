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
video_path1 = "../VideoFootage/rightCamMiddleCar.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("maskRightCamera1.png")

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
ParkingLot_A7 = [(1668,696),(1598,781),(1162,589),(1252,539)]
ParkingLot_A8 = [(1585,796),(1481,919),(1008,668),(1139,604)]
ParkingLot_A9 = [(1451,930),(1280,1080),(831,769),(988,681)]

ParkingLot_B1 = [(1297,256),(1281,272),(1201,228),(1235,217)]
ParkingLot_B2 = [(1240,263),(1215,290),(1142,255),(1173,232)]
ParkingLot_B3 = [(1183,292),(1166,316),(1083,276),(1105,250)]
ParkingLot_B4 = [(1125,317),(1090,342),(1017,287),(1053,269)]
ParkingLot_B5 = [(1047,328),(1015,356),(943,317),(962,274)]
ParkingLot_B6 = [(961,359),(908,380),(833,334),(870,304)]
ParkingLot_B7 = [(863,400),(821,444),(707,389),(758,344)]

ParkingLot_D1 = [(837,205),(795,216),(746,186),(787,172)]
ParkingLot_D2 = [(769,209),(725,217),(690,186),(727,180)]

#Store Location for each Parking Lot for Middle Camera
coordMidB = [490, 360]
ParkingLot_B8 = [(1822,coordMidB[0]),(1658,coordMidB[0]),(1524,coordMidB[1]),(1655,coordMidB[1])]
ParkingLot_B9 = [(1612,coordMidB[0]),(1444,coordMidB[0]),(1337,coordMidB[1]),(1491,coordMidB[1])]
ParkingLot_B10 = [(1401,coordMidB[0]),(1240,coordMidB[0]),(1178,coordMidB[1]),(1299,coordMidB[1])]
ParkingLot_B11 = [(1166,coordMidB[0]),(993,coordMidB[0]),(986,coordMidB[1]),(1134,coordMidB[1])]
ParkingLot_B12 = [(962,coordMidB[0]),(778,coordMidB[0]),(810,coordMidB[1]),(957,coordMidB[1])]
ParkingLot_B13 = [(738,coordMidB[0]),(570,coordMidB[0]),(642,coordMidB[1]),(778,coordMidB[1])]
ParkingLot_B14 = [(515,coordMidB[0]),(351,coordMidB[0]),(456,coordMidB[1]),(600,coordMidB[1])]
ParkingLot_B15 = [(287,coordMidB[0]),(130,coordMidB[0]),(292,coordMidB[1]),(419,coordMidB[1])]

coordMidC = [270, 230]
ParkingLot_C1 = [(1827,coordMidC[0]),(1774,coordMidC[0]),(1742,coordMidC[1]),(1816,coordMidC[1])]
ParkingLot_C2 = [(1705,coordMidC[0]),(1635,coordMidC[0]),(1614,coordMidC[1]),(1675,coordMidC[1])]
ParkingLot_C3 = [(1564,coordMidC[0]),(1502,coordMidC[0]),(1484,coordMidC[1]),(1533,coordMidC[1])]
ParkingLot_C4 = [(1401,coordMidC[0]),(1327,coordMidC[0]),(1312,coordMidC[1]),(1375,coordMidC[1])]
ParkingLot_C5 = [(1229,coordMidC[0]),(1167,coordMidC[0]),(1150,coordMidC[1]),(1216,coordMidC[1])]
ParkingLot_C6 = [(1091,coordMidC[0]),(1008,coordMidC[0]),(1012,coordMidC[1]),(1076,coordMidC[1])]
ParkingLot_C7 = [(934,coordMidC[0]),(864,coordMidC[0]),(861,coordMidC[1]),(937,coordMidC[1])]
ParkingLot_C8 = [(778,coordMidC[0]),(704,coordMidC[0]),(713,coordMidC[1]),(776,coordMidC[1])]
ParkingLot_C9 = [(632,coordMidC[0]),(558,coordMidC[0]),(580,coordMidC[1]),(647,coordMidC[1])]
ParkingLot_C10 = [(458,coordMidC[0]),(402,coordMidC[0]),(410,coordMidC[1]),(472,coordMidC[1])]
ParkingLot_C11 = [(297,coordMidC[0]),(232,coordMidC[0]),(241,coordMidC[1]),(306,coordMidC[1])]

coordMidD = [160, 125]
ParkingLot_D3 = [(1757,coordMidD[0]),(1702,coordMidD[0]),(1682,coordMidD[1]),(1738,coordMidD[1])]
ParkingLot_D4 = [(1655,coordMidD[0]),(1598,coordMidD[0]),(1571,coordMidD[1]),(1630,coordMidD[1])]
ParkingLot_D5 = [(1545,coordMidD[0]),(1498,coordMidD[0]),(1482,coordMidD[1]),(1537,coordMidD[1])]
ParkingLot_D6 = [(1444,coordMidD[0]),(1392,coordMidD[0]),(1380,coordMidD[1]),(1427,coordMidD[1])]
ParkingLot_D7 = [(1363,coordMidD[0]),(1309,coordMidD[0]),(1300,coordMidD[1]),(1345,coordMidD[1])]
ParkingLot_D8 = [(1263,coordMidD[0]),(1203,coordMidD[0]),(1205,coordMidD[1]),(1249,coordMidD[1])]
ParkingLot_D9 = [(1152,coordMidD[0]),(1101,coordMidD[0]),(1092,coordMidD[1]),(1143,coordMidD[1])]
ParkingLot_D10 = [(1045,coordMidD[0]),(1001,coordMidD[0]),(991,coordMidD[1]),(1037,coordMidD[1])]
ParkingLot_D11 = [(933,coordMidD[0]),(880,coordMidD[0]),(883,coordMidD[1]),(941,coordMidD[1])]
ParkingLot_D12 = [(843,coordMidD[0]),(777,coordMidD[0]),(785,coordMidD[1]),(848,coordMidD[1])]
ParkingLot_D13 = [(726,coordMidD[0]),(665,coordMidD[0]),(675,coordMidD[1]),(727,coordMidD[1])]
ParkingLot_D14 = [(604,coordMidD[0]),(557,coordMidD[0]),(564,coordMidD[1]),(607,coordMidD[1])]


# Camera box for left Camera
ParkingLot_A14 = [(1067,878),(836,1045),(491,999),(903,768)]
ParkingLot_A15 = [(877,754),(473,976),(362,852),(748,679)]
ParkingLot_A16 = [(730,667),(339,831),(223,711),(596,591)]
ParkingLot_A17 = [(577,581),(208,699),(148,630),(481,529)]


ParkingLot_B16 = [(1235,457),(1118,510),(1015,475),(1134,431)]
ParkingLot_B17 = [(1115,426),(996,472),(904,440),(1026,403)]
ParkingLot_B18 = [(1006,400),(893,437),(824,412),(942,381)]
ParkingLot_B19 = [(924,379),(816,409),(753,388),(866,362)]
ParkingLot_B20 = [(841,360),(736,387),(693,365),(795,341)]

ParkingLot_C12 = [(1227,366),(1173,410),(1093,391),(1192,339)]
ParkingLot_C13 = [(1147,358),(1072,383),(1009,370),(1111,338)]
ParkingLot_C14 = [(1060,343),(983,366),(932,352),(1020,320)]
ParkingLot_C15 = [(992,324),(915,343),(872,331),(947,304)]

ParkingLot_D15 = [(1520,298),(1513,342),(1462, 332),(1481, 298)]
ParkingLot_D16 = [(1432,296),(1422,326),(1372, 317),(1388, 292)]
ParkingLot_D17 = [(1336, 281),(1330, 316),(1290, 305),(1298, 281)]
ParkingLot_D18 = [(1271, 276),(1255, 300),(1208, 295),(1222, 275)]
ParkingLot_D19 = [(1203, 271),(1177, 296),(1137, 289),(1162, 270)]