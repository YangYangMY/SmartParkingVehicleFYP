from CarDetection.CarDetectController import *
from CarDetection.CarDetectionAlgorithm import *
from config import *


if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
print("Current time:", current_time)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Check if video files were opened successfully
if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
    print("Error opening video files")
    exit()

# Loop through the video frames
while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():

    # Read a frame from the video
    success1, frame1 = cap1.read()

    if success1:
        # Start the timer
        start1 = timeit.default_timer()

        imgRegion1 = cv2.bitwise_and(frame1, mask1)
        # Run YOLOv8 inference on the frame
        results = modelm(imgRegion1, stream=True, verbose=False)
        detections = np.empty((0,5))

        # Visualize the results on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "car" and conf > 0.3:
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(frame1,(RightCamEntryLine[0], RightCamEntryLine[1]), (RightCamEntryLine[2], RightCamEntryLine[3]),(0,0,255), 5)
        cv2.line(frame1, (RightCamMiddleLine[0], RightCamMiddleLine[1]),
                 (RightCamMiddleLine[2], RightCamMiddleLine[3]), (0, 0, 255), 5)
        cv2.line(frame1, (RightCamBottomLine[0], RightCamBottomLine[1]),
                 (RightCamBottomLine[2], RightCamBottomLine[3]), (0, 0, 255), 5)

        for results in resultsTracker:
            x1,y1,x2,y2,id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if (showCarID):
                cvzone.putTextRect(frame1, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
            if(showCarCenterDot):
                cv2.circle(frame1, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            if (showCarRecBox):
                cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=5)


            # Check if the car ID already exists in the dictionary
            if id in car_dict:
                # Update the bounding box coordinates for the existing car ID
                car_dict[id]["bbox"] = (x1, y1, x2, y2)
            else:
                # Add a new entry to the dictionary for the new car ID
                car_dict[id] = copy_car_dict.copy()
                car_dict[id]["bbox"] = (x1, y1, x2, y2)
                car_dict[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                car_dict[id]["currentLocation"] = "RightCam"

            if RightCamEntryLine[0] < cx < RightCamEntryLine[2] and RightCamEntryLine[1] > cy > RightCamEntryLine[3]:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(frame1, (RightCamEntryLine[0], RightCamEntryLine[1]), (RightCamEntryLine[2], RightCamEntryLine[3]), (0, 255, 0), 5)
                    # Mark the car ID as crossed in the car_dict
                    car_dict[id]["currentLocation"] = "RightCam"


            if RightCamBottomLine[0] < cx < RightCamBottomLine[2] and RightCamBottomLine[1] < cy < RightCamBottomLine[3]:
                cv2.line(frame1, (RightCamBottomLine[0], RightCamBottomLine[1]),
                         (RightCamBottomLine[2], RightCamBottomLine[3]), (0, 255, 0), 5)
                # Mark the car ID as crossed in the car_dict
                if id in car_temp and id in car_dict and car_dict[id]["currentLocation"] == "RightCam" and car_dict[id]["status"] == "moving":
                    # Update the bounding box coordinates for the existing car ID
                    car_temp[id] = car_dict[id]
                    if car_dict[id]["carPlate"] == "unknown":
                        car_dict[id]["carPlate"] = "test"
                else:
                    # Add a new entry to the dictionary for the new car ID
                    car_temp[id] = copy_car_dict.copy()
                    car_temp[id]["bbox"] = (x1, y1, x2, y2)
                    car_temp[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    car_temp[id]["currentLocation"] = "RightCam"

            if RightCamMiddleLine[0] < cx < RightCamMiddleLine[2]  and RightCamMiddleLine[1] < cy < RightCamMiddleLine[
                3]:
                cv2.line(frame1, (RightCamMiddleLine[0], RightCamMiddleLine[1]),
                         (RightCamMiddleLine[2], RightCamMiddleLine[3]), (0, 255, 0), 5)
                # Mark the car ID as crossed in the car_dict
                if id in car_temp3 and id in car_dict and car_dict[id]["currentLocation"] == "RightCam" and car_dict[id]["status"] == "moving":
                    # Update the bounding box coordinates for the existing car ID
                    car_temp3[id] = car_dict[id]
                    if car_dict[id]["carPlate"] == "unknown":
                        car_dict[id]["carPlate"] = "test"
                else:
                    # Add a new entry to the dictionary for the new car ID
                    for results2 in resultsTracker2:
                        x3, y3, x4, y4, id2 = results2
                        if id in car_temp3 and id2 in car_temp4:
                            if car_temp3[id]["entryTime"] != car_temp4[id2]["entryTime"]:
                                car_temp3[id] = copy_car_dict.copy()
                                car_temp3[id]["bbox"] = (x1, y1, x2, y2)
                                car_temp3[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                car_temp3[id]["currentLocation"] = "RightCam"



            #Parking Lot A1 - A20
            cv2.polylines(frame1, [np.array(ParkingLotCoordinateRightCam['D1'], np.int32)], True, (0, 0, 255), 2)
            resultsD1=cv2.pointPolygonTest(np.array(ParkingLotCoordinateRightCam['D1'],np.int32),((cx,cy)),False)
            if resultsD1>=0:
                if car_dict[id]["parkingLot"] == 'D1':
                    car_dict[id]["duration"] = (datetime.now() - car_dict[id]["parkingDetected"]).seconds
                    if car_dict[id]["duration"] > 30:
                        car_dict[id]["status"] = "parked"
                        if car_dict[id]["status"] == "parked":
                            ParkingLot['D1']["parked"] = "yes"
                            ParkingLot['D1']["carId"] = id
                        else:
                            ParkingLot['D1']["parked"] = "no"
                            ParkingLot['D1']["carId"] = "unknown"
                    else:
                        car_dict[id]["status"] = "moving"
                else:
                    car_dict[id]["parkingDetected"] = datetime.now()
                    car_dict[id]["parkingLot"] = 'D1'
                    ParkingLot['D1'] = {"carId": "unknown", "parked": "no"}
                    car_dict[id]["status"] = "moving"

            # Call the function with appropriate parameters
            process_parking(ParkingLotCoordinateRightCam, car_dict, id, cx, cy, 'D2')

            # draw Parking Box coordinate
            draw_parking_lot_polylines(frame1, ParkingLotCoordinateRightCam, showParkingLotBox)


        # Read a frame from the video
        success2, frame2 = cap2.read()

        if success2:
            # Start the timer
            start2 = timeit.default_timer()

            imgRegion2 = cv2.bitwise_and(frame2, mask2)
            # Run YOLOv8 inference on the frame
            results = modelx(imgRegion2, stream=True, verbose=False)

            detections2 = np.empty((0, 5))

            # Visualize the results on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])

                    currentClass = classNames[cls]

                    if currentClass == "car" and conf > 0.3:
                        currentArray2 = np.array([x1, y1, x2, y2, conf])
                        detections2 = np.vstack((detections2, currentArray2))

            resultsTracker2 = tracker2.update(detections2)

            cv2.line(frame2, (MiddleCamMiddleEntryLine[0], MiddleCamMiddleEntryLine[1]),
                     (MiddleCamMiddleEntryLine[2], MiddleCamMiddleEntryLine[3]), (0, 0, 255), 5)
            cv2.line(frame2, (MiddleCamMiddleExitLine[0], MiddleCamMiddleExitLine[1]),
                     (MiddleCamMiddleExitLine[2], MiddleCamMiddleExitLine[3]), (0, 0, 255), 5)
            cv2.line(frame2, (MiddleCamBottomEntryLine[0], MiddleCamBottomEntryLine[1]),
                     (MiddleCamBottomEntryLine[2], MiddleCamBottomEntryLine[3]), (0, 0, 255), 5)
            cv2.line(frame2, (MiddleCamBottomExitLine[0], MiddleCamBottomExitLine[1]),
                     (MiddleCamBottomExitLine[2], MiddleCamBottomExitLine[3]), (0, 0, 255), 5)

            for results in resultsTracker2:
                x1, y1, x2, y2, id = results
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                if (showCarID):
                    cvzone.putTextRect(frame2, f' {int(id)}', (max(0, x1), max(35, y1)),
                                     scale=2, thickness=3, offset=10)

                if (showCarCenterDot):
                    cv2.circle(frame2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if (showCarRecBox):
                    cvzone.cornerRect(frame2, (x1, y1, w, h), l=9, rt=5)

                # Check if the car ID already exists in the dictionary
                if id in car_dict:
                    # Update the bounding box coordinates for the existing car ID
                    car_dict[id]["bbox"] = (x1, y1, x2, y2)
                else:
                    # Add a new entry to the dictionary for the new car ID
                    car_dict[id] = copy_car_dict.copy()
                    car_dict[id]["bbox"] = (x1, y1, x2, y2)
                    car_dict[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    car_dict[id]["currentLocation"] = "MiddleCam"

                if MiddleCamMiddleEntryLine[0] < cx < MiddleCamMiddleEntryLine[2] and MiddleCamMiddleEntryLine[1] < cy < MiddleCamMiddleEntryLine[3]:
                    cv2.line(frame2, (MiddleCamMiddleEntryLine[0], MiddleCamMiddleEntryLine[1]),
                             (MiddleCamMiddleEntryLine[2], MiddleCamMiddleEntryLine[3]), (0, 255, 0), 5)
                    if id in car_temp4 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam":
                        car_temp4[id] = car_dict[id]
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp3 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "RightCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    del car_temp3[id2]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp4[id] = copy_car_dict.copy()
                        car_temp4[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp4[id]["currentLocation"] = "MiddleCam"
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp3 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "RightCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    del car_temp3[id2]

                if MiddleCamMiddleExitLine[0] < cx < MiddleCamMiddleExitLine[2] and MiddleCamMiddleExitLine[1] < cy < MiddleCamMiddleExitLine[3]:
                    cv2.line(frame2, (MiddleCamMiddleExitLine[0], MiddleCamMiddleExitLine[1]),
                             (MiddleCamMiddleExitLine[2], MiddleCamMiddleExitLine[3]), (0, 255, 0), 5)
                    if id in car_temp7 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam" and car_dict[id]["status"] == "moving":
                        car_temp7[id] = car_dict[id]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp7[id] = copy_car_dict.copy()
                        car_temp7[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp7[id]["currentLocation"] = "MiddleCam"

                if MiddleCamBottomEntryLine[0] > cx > MiddleCamBottomEntryLine[2] and MiddleCamBottomEntryLine[1] < cy < MiddleCamBottomEntryLine[3]:
                    cv2.line(frame2, (MiddleCamBottomEntryLine[0], MiddleCamBottomEntryLine[1]),
                             (MiddleCamBottomEntryLine[2], MiddleCamBottomEntryLine[3]), (0, 255, 0), 5)
                    if id in car_temp2 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam":
                        car_temp2[id] = car_dict[id]
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "RightCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "MiddleCam"
                                    del car_temp[id2]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp2[id] = copy_car_dict.copy()
                        car_temp2[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp2[id]["currentLocation"] = "MiddleCam"

                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "RightCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "MiddleCam"
                                    del car_temp[id2]

                if MiddleCamBottomExitLine[0] > cx > MiddleCamBottomExitLine[2] and MiddleCamBottomExitLine[1] < cy < MiddleCamBottomExitLine[3]:
                    cv2.line(frame2, (MiddleCamBottomExitLine[0], MiddleCamBottomExitLine[1]),
                             (MiddleCamBottomExitLine[2], MiddleCamBottomExitLine[3]), (0, 255, 0), 5)
                    if id in car_temp5 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam" and car_dict[id]["status"] == "moving":
                        car_temp5[id] = car_dict[id]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp5[id] = copy_car_dict.copy()
                        car_temp5[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp5[id]["currentLocation"] = "MiddleCam"



            #draw Parking Box coordinate
            draw_parking_lot_polylines(frame2, ParkingLotCoordinateMidCam, showParkingLotBox)

            # Read a frame from the video
            success3, frame3 = cap3.read()

        if success3:
            # Start the timer
            start3 = timeit.default_timer()

            imgRegion3 = cv2.bitwise_and(frame3, mask3)
            # Run YOLOv8 inference on the frame
            results = modelm(imgRegion3, stream=True, verbose=False)

            detections3 = np.empty((0, 5))

            # Visualize the results on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])

                    currentClass = classNames[cls]

                    if currentClass == "car" and conf > 0.3:
                        currentArray3 = np.array([x1, y1, x2, y2, conf])
                        detections3 = np.vstack((detections3, currentArray3))

            resultsTracker3 = tracker3.update(detections3)
            cv2.line(frame3, (LeftCamMiddleEntryLine[0], LeftCamMiddleEntryLine[1]),
                     (LeftCamMiddleEntryLine[2], LeftCamMiddleEntryLine[3]), (0, 0, 255), 5)
            cv2.line(frame3, (LeftCamMiddleExitLine[0], LeftCamMiddleExitLine[1]),
                     (LeftCamMiddleExitLine[2], LeftCamMiddleExitLine[3]), (0, 0, 255), 5)

            cv2.line(frame3, (LeftCamBottomEntryLine[0], LeftCamBottomEntryLine[1]),
                     (LeftCamBottomEntryLine[2], LeftCamBottomEntryLine[3]), (0, 0, 255), 5)
            cv2.line(frame3, (LeftCamBottomExitLine[0], LeftCamBottomExitLine[1]),
                     (LeftCamBottomExitLine[2], LeftCamBottomExitLine[3]), (0, 0, 255), 5)

            for results in resultsTracker3:
                x1, y1, x2, y2, id = results
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                if (showCarID):
                    cvzone.putTextRect(frame3, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

                if (showCarCenterDot):
                    cv2.circle(frame3, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if (showCarRecBox):
                    cvzone.cornerRect(frame3, (x1, y1, w, h), l=9, rt=5)

                # Check if the car ID already exists in the dictionary
                if id in car_dict:
                    # Update the bounding box coordinates for the existing car ID
                    car_dict[id]["bbox"] = (x1, y1, x2, y2)
                else:
                    # Add a new entry to the dictionary for the new car ID
                    car_dict[id] = copy_car_dict.copy()
                    car_dict[id]["bbox"] = (x1, y1, x2, y2)
                    car_dict[id]["currentLocation"] = "LeftCam"

                if LeftCamMiddleEntryLine[0] > cx > LeftCamMiddleEntryLine[2] and LeftCamMiddleEntryLine[1] < cy < LeftCamMiddleEntryLine[3]:
                    if id not in car_temp8:
                        cv2.line(frame3, (LeftCamMiddleEntryLine[0], LeftCamMiddleEntryLine[1]),
                             (LeftCamMiddleEntryLine[2], LeftCamMiddleEntryLine[3]), (0, 255, 0), 5)
                    if id in car_temp8 and id in car_dict and car_dict[id]["currentLocation"] == "LeftCam":
                        car_temp8[id] = copy_car_dict.copy()
                        car_temp8[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp8[id]["currentLocation"] = "LeftCam"
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker2:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp7 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "MiddleCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "LeftCam"
                                    del car_temp7[id2]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp8[id] = copy_car_dict.copy()
                        car_temp8[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp8[id]["currentLocation"] = "LeftCam"
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker2:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp7 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "MiddleCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "LeftCam"
                                    del car_temp7[id2]

                if LeftCamBottomEntryLine[0] > cx > LeftCamBottomEntryLine[2] and LeftCamBottomEntryLine[1] < cy < LeftCamBottomEntryLine[3]:
                    cv2.line(frame3, (LeftCamBottomEntryLine[0], LeftCamBottomEntryLine[1]),
                             (LeftCamBottomEntryLine[2], LeftCamBottomEntryLine[3]), (0, 255, 0), 5)
                    if id in car_temp6 and id in car_dict and car_dict[id]["currentLocation"] == "LeftCam":
                        car_temp6[id] = car_dict[id]
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker2:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp5 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "MiddleCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "LeftCam"
                                    del car_temp5[id2]
                    else:
                        # Add a new entry to the dictionary for the new car ID
                        car_temp6[id] = copy_car_dict.copy()
                        car_temp6[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp6[id]["currentLocation"] = "LeftCam"
                        if car_dict[id]["carPlate"] == "unknown":
                            for results in resultsTracker2:
                                x1, y1, x2, y2, id2 = results
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                if id2 in car_temp5 and id2 in car_dict and car_dict[id2]["status"] == "moving" and car_dict[id2]["currentLocation"] == "MiddleCam":
                                    car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                    car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                    car_dict[id2]["currentLocation"] = "LeftCam"
                                    del car_temp5[id2]

                if LeftCamBottomExitLine[0] < cx < LeftCamBottomExitLine[2] and LeftCamBottomExitLine[1] > cy > LeftCamBottomExitLine[3] :
                    cv2.line(frame3, (LeftCamBottomExitLine[0], LeftCamBottomExitLine[1]),
                                 (LeftCamBottomExitLine[2], LeftCamBottomExitLine[3]), (0, 255, 0), 5)
                    if id in car_dict:
                        car_dict[id]["exitTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                if LeftCamMiddleExitLine[0] > cx > LeftCamMiddleExitLine[2] and LeftCamMiddleExitLine[1] < cy < LeftCamMiddleExitLine[3]:
                    cv2.line(frame3, (LeftCamMiddleExitLine[0], LeftCamMiddleExitLine[1]),
                                 (LeftCamMiddleExitLine[2], LeftCamMiddleExitLine[3]), (0, 255, 0), 5)
                    if id in car_dict:
                        car_dict[id]["exitTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                draw_parking_lot_polylines(frame3, ParkingLotCoordinateLeftCam, showParkingLotBox)

        print(car_dict)
        print(ParkingLot['B8'])
        #print(car_dict)
        # Bottom Lane
        #print("1:", car_temp)  # Right Cam Exit
        #print("2:", car_temp2)  # middle Cam Entry
        #print("5:", car_temp5)  # Middle Cam Exit
        #print("6:", car_temp6)  # Left Cam Entry

        # Middle Lane
        #print("3:", car_temp3)  # Right Cam Exit
        #print("4:", car_temp4)  # Middle Cam Entry
        #print("7:", car_temp7)  # Middle Cam Exit
        #print("8:", car_temp8)  # Left Cam Entry

        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame1 = cv2.resize(frame1, (800, 600))
        resized_frame2 = cv2.resize(frame2, (800, 600))
        resized_frame3 = cv2.resize(frame3, (800, 600))

        # Display the frame rate on the left top of the screen
        #cv2.putText(resized_frame1, f' Count: {len(totalCount)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Stop the timer and calculate the FPS
        end = timeit.default_timer()
        yolo_fps1 = 1 / (end - start1)
        yolo_fps2 = 1 / (end - start2)
        yolo_fps3 = 1 / (end - start3)

        # Display the frame rate on the left top of the screen
        # cv2.putText(resized_frame1, "FPS: " + "{:.2f}".format(yolo_fps1), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)
       # cv2.putText(resized_frame2, "FPS: " + "{:.2f}".format(yolo_fps2), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
       # cv2.putText(resized_frame3, "FPS: " + "{:.2f}".format(yolo_fps3), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
         #           (0, 0, 255), 2)

        # Combine the frames side by side
        combined_frame = cv2.hconcat([resized_frame2, resized_frame1])

        # Display the combined frame
       # cv2.imshow("temp1", resized_frame1)
        #cv2.imshow("Video 2", resized_frame2)
        #cv2.imshow("Video 3", resized_frame3)

        #Display Combine frame
        cv2.imshow("Combined Video", combined_frame)
        cv2.imshow("Video 3", resized_frame3)

        #Display frame for coordinate
        #cv2.imshow("temp1", frame2)
        #cv2.imshow("temp2", frame2)
        #cv2.imshow("temp1", frame3)

        #cv2.setMouseCallback('temp1', RGB)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()