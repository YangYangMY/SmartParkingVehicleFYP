import cv2

# This File will be a controller to control all possible codes for car detection

#Parking Line for passing data when car go through next camera
showParkingLotLine = True

#Parking Box
showParkingLotBox = True
showDoubleParkingLotBox = True

#Car Settings
showCarID = True
showCarCenterDot = True
showCarRecBox = False

#Car Detection Settings
PARKING_TIME = 30  # seconds to determine if a car is parked or not
DOUBLEPARK_TIME = 30  # seconds to determine if a car is double park or not


#Output Video Settings
showResizedVideo = True
showCombinedVideo = True #If this is enabled, showLeftCam & showMidCam & showRightCam won't work
showLeftCam = True
showMidCam = False
showRightCam = False
showFPS = False

#Video File Settings
# Open the video file 1
video_path1 = "VideoFootage/rightCam2.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("MaskImage/maskRightCamera.png")

# Open the video file 2
video_path2 = "VideoFootage/midCam2.mp4"
cap2 = cv2.VideoCapture(video_path2)
mask2 = cv2.imread("MaskImage/maskMiddleCamera.png")

# Open the video file 3
video_path3 = "VideoFootage/leftCam2.mp4"
cap3 = cv2.VideoCapture(video_path3)
mask3 = cv2.imread("MaskImage/maskLeftCamera.png")




