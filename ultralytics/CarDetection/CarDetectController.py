# This File will be a controller to control all possible codes for car detection
import cv2

from ultralytics import YOLO

#Parking Line for passing data when car go through next camera
showParkingLotLine = True

#Parking Box
showParkingLotBox = True

#Car Settings
showCarID = True
showCarCenterDot = True
showCarRecBox = False

#Car Detection Settings
PARKING_TIMEOUT = 30  # seconds to determine if a car is parked or not

#Output Video Settings
showResizedCombinedVideo = False
showFPS = False

# Open the video file 1
video_path1 = "../VideoFootage/rightCam3.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("MaskImage/maskRightCamera.png")

# Open the video file 2
video_path2 = "../VideoFootage/midCam3.mp4"
cap2 = cv2.VideoCapture(video_path2)
mask2 = cv2.imread("MaskImage/maskMiddleCamera.png")

# Open the video file 3
video_path3 = "../VideoFootage/leftCam3.mp4"
cap3 = cv2.VideoCapture(video_path3)
mask3 = cv2.imread("MaskImage/maskLeftCamera.png")

# Load the YOLO v8 pre-trained model
modelm = YOLO('../Models/yolov8m.pt')
modelx = YOLO('../Models/yolov8x.pt')


#Print Car Dictionary Data
# print(car_dict)
# print(parking_lots['B8'])
# print(car_dict)
# Bottom Lane
# print("1:", car_temp)  # Right Cam Exit
# print("2:", car_temp2)  # middle Cam Entry
# print("5:", car_temp5)  # Middle Cam Exit
# print("6:", car_temp6)  # Left Cam Entry

# Middle Lane
# print("3:", car_temp3)  # Right Cam Exit
# print("4:", car_temp4)  # Middle Cam Entry
# print("7:", car_temp7)  # Middle Cam Exit
# print("8:", car_temp8)  # Left Cam Entry