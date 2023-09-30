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
DOUBLE_PARK_TIME = 30  # seconds to determine if a car is double park or not

#Microsoft Excel Settings
export_excel = True  # export data to Google Sheet
export_excel_time = 5  # seconds to export data to Google Sheet
ExportExcel_filename = "CarDetectionData"
car_dict_column_names = ["carId", "carPlate", "bbox", "entryTime", "exitTime", "status", "currentLocation",
                "parkingLot", "parkingDetected", "duration", "isDoubleParked", "doubleParkingLot", "isDoubleParking"]
double_park_lots_column_names = ["parkingLot", "carId", "parked"]
parking_lots_column_names = ["parkingLot", "carId", "parked"]

#Output Video Settings
common_frame_rate = 5  # Adjust as needed
showParkingOccupancy = True
showResizedVideo = True
showLeftCam = True
showMidCam = True
showRightCam = True
showFPS = True

#Video File Settings
# Open the video file 1
video_path1 = "CarDetection/VideoFootage/rightCamera1080.mp4"
cap1 = cv2.VideoCapture(video_path1)
mask1 = cv2.imread("CarDetection/MaskImage/maskRightCamera.png")

# Open the video file 2
video_path2 = "CarDetection/VideoFootage/middleCamera1080.mp4"
cap2 = cv2.VideoCapture(video_path2)
mask2 = cv2.imread("CarDetection/MaskImage/maskMiddleCamera.png")

# Open the video file 3
video_path3 = "CarDetection/VideoFootage/leftCamera1080.mp4"
cap3 = cv2.VideoCapture(video_path3)
mask3 = cv2.imread("CarDetection/MaskImage/maskLeftCamera.png")




