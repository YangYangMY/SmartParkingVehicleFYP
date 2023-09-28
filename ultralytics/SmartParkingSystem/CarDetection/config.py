from ultralytics import YOLO
from SmartParkingSystem.CarDetection.sort import *

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

# Load the YOLO v8 pre-trained model
modelm = YOLO('CarDetection/Models/yolov8m.pt')
modelx = YOLO('CarDetection/Models/yolov8x.pt')
modelx2 = YOLO('CarDetection/Models/yolov8x.pt')

#Store Car Dictionary
copy_car_dict = {"carPlate": "-" ,"bbox": "-", "entryTime": "-",
                 "exitTime": "-", "status": "moving", "currentLocation": "-",
                 "parkingLot": "-", "parkingDetected": "-", "duration": "-",
                 "isDoubleParked": "no", "doubleParkingLot": "-", "isDoubleParking": "no"}
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
parking_lots = {}
parking_lots_empty = {"carId": "-", "parked": "no"}

lotA_range = [7, 8, 9, 14, 15, 16, 17, 18]
for number in lotA_range:
    lot_name = f'A{number}'
    parking_lots[lot_name] = parking_lots_empty.copy()

for i in range(20):
    i += 1
    parking_lots['D'+ str(i)] = parking_lots_empty.copy()
    parking_lots['B'+ str(i)] = parking_lots_empty.copy()

for i in range(15):
    i += 1
    parking_lots['C'+ str(i)] = parking_lots_empty.copy()

# Store each Parking lot Coordinate
ParkingLotCoordinateRightCam = {}

# Parking Lot Data Right Cam
parking_lots_data_RightCam = {
    'A7': [(1668,696),(1598,781),(1162,589),(1252,539)],
    'A8': [(1585,796),(1481,919),(1008,668),(1139,604)],
    'A9': [(1451,930),(1280,1080),(831,769),(988,681)],
    'B1': [(1297,256),(1281,272),(1201,228),(1235,217)],
    'B2': [(1240,263),(1215,290),(1142,255),(1173,232)],
    'B3': [(1183,292),(1166,316),(1083,276),(1105,250)],
    'B4': [(1125,317),(1090,342),(1017,287),(1053,269)],
    'B5': [(1047,328),(1015,356),(943,317),(962,274)],
    'B6': [(961,359),(908,380),(833,334),(870,304)],
    'B7': [(863,400),(809, 436),(707,389),(758,344)],
    'D1': [(837,205),(795,216),(746,186),(787,172)],
    'D2': [(769,209),(725,217),(690,186),(727,180)]
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in parking_lots_data_RightCam.items():
    ParkingLotCoordinateRightCam[lot_name] = coordinates


#Store Location for each Parking Lot for Middle Camera
ParkingLotCoordinateMidCam = {}
coordMidB = [490, 350]
coordMidC = [270, 230]
coordMidD = [160, 125]

# Create the parking_lots_data_mid_cam dictionary with the original format
parking_lots_data_mid_cam = {
    'B8': [(1844, coordMidB[0]), (1658, coordMidB[0]), (1524, coordMidB[1]), (1704, coordMidB[1])],
    'B9': [(1629, coordMidB[0]), (1444, coordMidB[0]), (1337, coordMidB[1]), (1491, coordMidB[1])],
    'B10': [(1411, coordMidB[0]), (1204, coordMidB[0]), (1154, coordMidB[1]), (1313, coordMidB[1])],
    'B11': [(1191, coordMidB[0]), (993, coordMidB[0]), (986, coordMidB[1]), (1134, coordMidB[1])],
    'B12': [(962, coordMidB[0]), (778, coordMidB[0]), (810, coordMidB[1]), (957, coordMidB[1])],
    'B13': [(738, coordMidB[0]), (543, coordMidB[0]), (642, coordMidB[1]), (778, coordMidB[1])],
    'B14': [(515, coordMidB[0]), (326, coordMidB[0]), (456, coordMidB[1]), (600, coordMidB[1])],
    'B15': [(287, coordMidB[0]), (112, coordMidB[0]), (292, coordMidB[1]), (419, coordMidB[1])],
    'C1': [(1827, coordMidC[0]), (1774, coordMidC[0]), (1742, coordMidC[1]), (1816, coordMidC[1])],
    'C2': [(1705, coordMidC[0]), (1635, coordMidC[0]), (1614, coordMidC[1]), (1675, coordMidC[1])],
    'C3': [(1564, coordMidC[0]), (1502, coordMidC[0]), (1484, coordMidC[1]), (1533, coordMidC[1])],
    'C4': [(1401, coordMidC[0]), (1327, coordMidC[0]), (1312, coordMidC[1]), (1375, coordMidC[1])],
    'C5': [(1229, coordMidC[0]), (1167, coordMidC[0]), (1150, coordMidC[1]), (1216, coordMidC[1])],
    'C6': [(1091, coordMidC[0]), (1008, coordMidC[0]), (1012, coordMidC[1]), (1076, coordMidC[1])],
    'C7': [(934, coordMidC[0]), (864, coordMidC[0]), (861, coordMidC[1]), (937, coordMidC[1])],
    'C8': [(778, coordMidC[0]), (704, coordMidC[0]), (713, coordMidC[1]), (776, coordMidC[1])],
    'C9': [(632, coordMidC[0]), (558, coordMidC[0]), (580, coordMidC[1]), (647, coordMidC[1])],
    'C10': [(458, coordMidC[0]), (402, coordMidC[0]), (410, coordMidC[1]), (472, coordMidC[1])],
    'C11': [(297, coordMidC[0]), (232, coordMidC[0]), (241, coordMidC[1]), (306, coordMidC[1])],
    'D3': [(1757, coordMidD[0]), (1702, coordMidD[0]), (1682, coordMidD[1]), (1738, coordMidD[1])],
    'D4': [(1655, coordMidD[0]), (1598, coordMidD[0]), (1571, coordMidD[1]), (1630, coordMidD[1])],
    'D5': [(1545, coordMidD[0]), (1498, coordMidD[0]), (1482, coordMidD[1]), (1537, coordMidD[1])],
    'D6': [(1444, coordMidD[0]), (1392, coordMidD[0]), (1380, coordMidD[1]), (1427, coordMidD[1])],
    'D7': [(1363, coordMidD[0]), (1309, coordMidD[0]), (1300, coordMidD[1]), (1345, coordMidD[1])],
    'D8': [(1263, coordMidD[0]), (1203, coordMidD[0]), (1205, coordMidD[1]), (1249, coordMidD[1])],
    'D9': [(1152, coordMidD[0]), (1101, coordMidD[0]), (1092, coordMidD[1]), (1143, coordMidD[1])],
    'D10': [(1045, coordMidD[0]), (1001, coordMidD[0]), (991, coordMidD[1]), (1037, coordMidD[1])],
    'D11': [(933, coordMidD[0]), (880, coordMidD[0]), (883, coordMidD[1]), (941, coordMidD[1])],
    'D12': [(843, coordMidD[0]), (777, coordMidD[0]), (785, coordMidD[1]), (848, coordMidD[1])],
    'D13': [(726, coordMidD[0]), (665, coordMidD[0]), (675, coordMidD[1]), (727, coordMidD[1])],
    'D14': [(604, coordMidD[0]), (557, coordMidD[0]), (564, coordMidD[1]), (607, coordMidD[1])],
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in parking_lots_data_mid_cam.items():
    ParkingLotCoordinateMidCam[lot_name] = coordinates


# Camera box for left Camera
ParkingLotCoordinateLeftCam = {}

parking_lots_data_left_cam = {
    'A14': [(1067, 878), (800, 1080), (491, 999), (903, 768)],
    'A15': [(877, 754), (473, 976), (362, 852), (748, 679)],
    'A16': [(730, 667), (339, 831), (223, 711), (596, 591)],
    'A17': [(577, 581), (208, 699), (148, 630), (481, 529)],
    'A18': [(470, 522), (138, 621), (73, 557), (398, 477)],
    'B16': [(1235, 457), (1118, 510), (1015, 475), (1134, 431)],
    'B17': [(1115, 426), (996, 472), (904, 440), (1026, 403)],
    'B18': [(1006, 400), (893, 437), (824, 412), (942, 381)],
    'B19': [(924, 379), (816, 409), (753, 388), (866, 362)],
    'B20': [(841, 360), (736, 387), (693, 365), (795, 341)],
    'C12': [(1227, 366), (1173, 410), (1093, 391), (1192, 339)],
    'C13': [(1147, 358), (1072, 383), (1009, 370), (1111, 338)],
    'C14': [(1060, 343), (983, 366), (932, 352), (1020, 320)],
    'C15': [(992, 324), (915, 343), (872, 331), (947, 304)],
    'D15': [(1520, 298), (1513, 342), (1462, 332), (1481, 298)],
    'D16': [(1432, 296), (1422, 326), (1372, 317), (1388, 292)],
    'D17': [(1336, 281), (1330, 316), (1290, 305), (1298, 281)],
    'D18': [(1271, 276), (1255, 300), (1208, 295), (1222, 275)],
    'D19': [(1203, 271), (1177, 296), (1137, 289), (1162, 270)]
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in parking_lots_data_left_cam.items():
    ParkingLotCoordinateLeftCam[lot_name] = coordinates

# Store each Double Park lot Coordinate
double_park_lots = {}
double_park_lots_empty = {"carId": "-", "parked": "no", "covered_parking_lot": []}

DoubleParkCoordinateRightCam = {}

# Parking Lot Data Right Cam
double_park_data_RightCam = {
    'T1': [(1350, 281),(1468, 314),(1412, 334),(1300, 294)],
    'T2': [(1300, 294),(1412, 334),(1360, 356),(1243, 314)],
    'T3': [(1243, 314),(1360, 356),(1301, 382),(1180, 330)],
    'T4': [(1180, 330),(1301, 382),(1238, 405),(1101, 353)],
    'T5': [(1101, 353),(1238, 405),(1162, 431),(1015, 380)],
    'T6': [(1015, 380),(1162, 431),(1067, 467),(923, 410)],
    'T7': [(923, 410),(1067, 467),(802, 572),(660, 491)],
    'DB7': [(1079, 471),(1242, 524),(1129, 584),(972, 515)],
    'DB8': [(972, 515),(1129, 584),(984, 657),(813, 578)],
    'DB9': [(813, 578),(984, 657),(785, 758),(625, 654)],
    'DB10': [(625, 654),(785, 758),(514, 900),(380, 755)],
}

double_park_covered_lots_RightCam = {
    'T1': ["B1", "B2"],
    'T2': ["B1", "B2","B3"],
    'T3': ["B2", "B3", "B4"],
    'T4': ["B3", "B4", "B5"],
    'T5': ["B4", "B5", "B6"],
    'T6': ["B5", "B6", "B7"],
    'T7': ["B6", "B7", "B8"],
    'DB7': ["A7", "A8"],
    'DB8': ["A7", "A8", "A9"],
    'DB9': ["A8", "A9"],
    'DB10': ["A9"],
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in double_park_data_RightCam.items():
    DoubleParkCoordinateRightCam[lot_name] = coordinates

# Store each Double Park lot Coordinate
for lot_name in double_park_data_RightCam.keys():
    double_park_lots[lot_name] = double_park_lots_empty.copy()
    covered_lots = list(double_park_covered_lots_RightCam[lot_name])
    double_park_lots[lot_name]["covered_parking_lot"] = sorted(covered_lots)


DoubleParkCoordinateMidCam = {}

# Parking Lot Data Right Cam
double_park_data_MidCam = {
    'T8': [(1882, 515),(1918, 661),(1755, 658),(1663, 511)],
    'T9': [(1663, 511),(1755, 658),(1513, 660),(1439, 513)],
    'T10': [(1439, 513),(1513, 660),(1253, 659),(1206, 512)],
    'T11': [(1206, 512),(1253, 659),(993, 660),(980, 512)],
    'T12': [(980, 512),(993, 660),(727, 660),(750, 510)],
    'T13': [(750, 510),(727, 660),(457, 656),(522, 510)],
    'T14': [(522, 510),(457, 656),(189, 657),(287, 513)],
    'T15': [(287, 513),(189, 657),(1, 639),(70, 511)],
}

double_park_covered_lots_MidCam = {
    'T8': ["B7", "B8", "B9"],
    'T9': ["B8","B9", "B10"],
    'T10': ["B9", "B10", "B11"],
    'T11': ["B10", "B11", "B12"],
    'T12': ["B11", "B12", "B13"],
    'T13': ["B12", "B13", "B14"],
    'T14': ["B13", "B14", "B15"],
    'T15': ["B14", "B15", "B16"],
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in double_park_data_MidCam.items():
    DoubleParkCoordinateMidCam[lot_name] = coordinates

# Store each Double Park lot Coordinate
for lot_name in double_park_data_MidCam.keys():
    double_park_lots[lot_name] = double_park_lots_empty.copy()
    covered_lots = list(double_park_covered_lots_MidCam[lot_name])
    double_park_lots[lot_name]["covered_parking_lot"] = sorted(covered_lots)


DoubleParkCoordinateLeftCam = {}

# Parking Lot Data Left Cam
double_park_data_LeftCam = {
    'T16': [(1072, 533),(944, 588),(817, 534),(945, 491)],
    'T17': [(945, 491),(817, 534),(718, 494),(843, 456)],
    'T18': [(843, 456),(718, 494),(640, 461),(758, 425)],
    'T19': [(758, 425),(640, 461),(561, 426),(678, 400)],
    'T20': [(678, 400),(561, 426),(493, 398),(614, 378)],
    'DB14': [(1131, 864),(1310, 753),(1082, 658),(913, 747)],
    'DB15': [(913, 747),(1082, 658),(931, 596),(757, 663)],
    'DB16': [(757, 663),(931, 596),(774, 526),(617, 585)],
    'DB17': [(617, 585),(774, 526),(673, 482),(507, 523)],
    'DB18': [(507, 523),(673, 482),(567, 435),(420, 471)],
}

double_park_covered_lots_LeftCam = {
    'T16': ["B15", "B16", "B17"],
    'T17': ["B16", "B17", "B18"],
    'T18': ["B17", "B18", "B19"],
    'T19': ["B18", "B19", "B20"],
    'T20': ["B19","B20"],
    'DB14': ["A14", "A15"],
    'DB15': ["A14", "A15", "A16"],
    'DB16': ["A15", "A16", "A17"],
    'DB17': ["A16", "A17", "A18"],
    'DB18': ["A17","A18"],
}

# Convert parking lot data to the desired format and store in the ParkingLotCoordinate dictionary
for lot_name, coordinates in double_park_data_LeftCam.items():
    DoubleParkCoordinateLeftCam[lot_name] = coordinates

# Store each Double Park lot Coordinate
for lot_name in double_park_data_LeftCam.keys():
    double_park_lots[lot_name] = double_park_lots_empty.copy()
    covered_lots = list(double_park_covered_lots_LeftCam[lot_name])
    double_park_lots[lot_name]["covered_parking_lot"] = sorted(covered_lots)

