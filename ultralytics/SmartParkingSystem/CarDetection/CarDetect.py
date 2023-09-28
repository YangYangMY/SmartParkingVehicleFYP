import math
import timeit

import cvzone

from SmartParkingSystem.CarDetection.CarDetectController import *
from SmartParkingSystem.CarDetection.CarDetectVideoOutput import ShowVideoOutput
from SmartParkingSystem.CarDetection.CarDetectionAlgorithm import *
from SmartParkingSystem.CarDetection.MicrosoftExcelIntergration import update_excel_with_car_dict, update_excel_with_parking_lots, \
    update_excel_with_double_parking_lots
from SmartParkingSystem.CarDetection.config import *
import xlwings as xw
import logging

# Create a shared variable to control the expo
# rt thread
stop_all_func = False
# Initialize the Excel application (outside the function)
app = xw.apps.active


# Function to export data periodically
def export_data_to_excel_thread(ExportExcel_filename, formatted_current_time, car_dict_column_names, car_dict, parking_lots_column_names, parking_lots, double_park_lots_column_names, double_park_lots):
    try:
        global app  # Use the globally defined Excel application instance
        global stop_all_func  # Set the flag to stop the thread

        if app is None:
            app = xw.App(visible=True)  # Start a new instance if Excel is not running

        while not stop_all_func:  # Check the flag in each iteration
            try:
                # Get the current date and time for creating a new filename
                filename = f"{ExportExcel_filename}_{formatted_current_time}.xlsx"
                full_path = os.path.join("CarDetection/OutputData", filename)

                # Check if the file already exists
                if os.path.exists(full_path):
                    logging.warning(f"File '{full_path}' already exists. It will be overwritten.")

                # Open the Excel workbook or create a new one
                wb = app.books.open(full_path) if os.path.exists(full_path) else app.books.add()

                # Store data in the sheet
                update_excel_with_double_parking_lots(app, wb, filename, full_path, "Double Parking Lots", double_park_lots_column_names, double_park_lots)
                update_excel_with_parking_lots(app, wb, filename, full_path, "Parking Lots", parking_lots_column_names, parking_lots)
                update_excel_with_car_dict(app, wb, filename, full_path, "Car Data", car_dict_column_names, car_dict)

                # Save the workbook (this keeps it open)
                wb.save(full_path)

            except Exception as e:
                logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")

            time.sleep(export_excel_time + 10)  # Adjust sleep time as needed

    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")
    finally:
        if app is not None:
            stop_all_func = True  # Set the flag to stop the thread
            app.quit()  # Close the Excel application

# Main video processing function
def process_video():
    global stop_all_func  # Access the shared variable

    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        # Read a frame from the video
        success1, frame1 = cap1.read()

        if success1 and frame1 is not None and stop_all_func is False:
            # Start the timer
            start1 = timeit.default_timer()

            imgRegion1 = cv2.bitwise_and(frame1, mask1)

            # Run YOLOv8 inference on the frame
            results = modelm(imgRegion1, stream=True, verbose=False)
            detections = np.empty((0, 5))

            # Visualize the results on the frame
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
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)

            if showParkingLotLine:
                for line in [RightCamEntryLine, RightCamMiddleLine, RightCamBottomLine]:
                    cv2.line(frame1, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)

            for results in resultsTracker:
                x1, y1, x2, y2, id = map(int, results)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                if showCarID:
                    cvzone.putTextRect(frame1, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                                       offset=10)
                if showCarCenterDot:
                    cv2.circle(frame1, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                if showCarRecBox:
                    cvzone.cornerRect(frame1, (x1, y1, w, h), l=9, rt=5)

                update_car_info(id, x1, y1, x2, y2, car_dict, "RightCam")

                if check_car_cross_line1(RightCamEntryLine, cx, cy):
                    if totalCount.count(id) == 0:
                        totalCount.append(id)
                        cv2.line(frame1, (RightCamEntryLine[0], RightCamEntryLine[1]),
                                 (RightCamEntryLine[2], RightCamEntryLine[3]), (0, 255, 0), 5)
                        car_dict[id]["currentLocation"] = "RightCam"

                if check_car_cross_line2(RightCamBottomLine, cx, cy):
                    cv2.line(frame1, (RightCamBottomLine[0], RightCamBottomLine[1]),
                             (RightCamBottomLine[2], RightCamBottomLine[3]), (0, 255, 0), 5)
                    if id in car_temp and id in car_dict and car_dict[id]["currentLocation"] == "RightCam" and car_dict[
                        id].get("status") == "moving":
                        car_temp[id] = car_dict[id]
                        if car_dict[id].get("carPlate") == "-":
                            car_dict[id]["carPlate"] = "test"
                    else:
                        car_temp[id] = copy_car_dict.copy()
                        car_temp[id]["bbox"] = (x1, y1, x2, y2)
                        car_temp[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                        car_temp[id]["currentLocation"] = "RightCam"

                if check_car_cross_line2(RightCamMiddleLine, cx, cy):
                    cv2.line(frame1, (RightCamMiddleLine[0], RightCamMiddleLine[1]),
                             (RightCamMiddleLine[2], RightCamMiddleLine[3]), (0, 255, 0), 5)
                    if id in car_temp3 and id in car_dict and car_dict[id]["currentLocation"] == "RightCam" and \
                            car_dict[id]["status"] == "moving":
                        car_temp3[id] = car_dict[id]
                        if car_dict[id]["carPlate"] == "-":
                            car_dict[id]["carPlate"] = "test"
                    else:
                        for results2 in resultsTracker2:
                            x3, y3, x4, y4, id2 = map(int, results2)
                            if id in car_temp3 and id2 in car_temp4 and car_temp3[id]["entryTime"] != car_temp4[id2][
                                "entryTime"]:
                                car_temp3[id] = copy_car_dict.copy()
                                car_temp3[id]["bbox"] = (x1, y1, x2, y2)
                                car_temp3[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                car_temp3[id]["currentLocation"] = "RightCam"

                draw_parking_lot_polylines(frame1, ParkingLotCoordinateRightCam, showParkingLotBox, parking_lots,
                                           car_dict)
                draw_parking_lot_polylines(frame1, DoubleParkCoordinateRightCam, showDoubleParkingLotBox,
                                           double_park_lots, car_dict)

                for parking_lot_name in ParkingLotCoordinateRightCam.keys():
                    process_parking(ParkingLotCoordinateRightCam, car_dict, id, cx, cy, parking_lot_name, parking_lots)

                for parking_lot_name in DoubleParkCoordinateRightCam.keys():
                    process_double_parking(DoubleParkCoordinateRightCam, car_dict, id, cx, cy, parking_lot_name,
                                           double_park_lots, parking_lots)

            # Read a frame from the video
            success2, frame2 = cap2.read()

            if success2 and frame2 is not None and stop_all_func is False:
                # Start the timer
                start2 = timeit.default_timer()

                imgRegion2 = cv2.bitwise_and(frame2, mask2)

                # Run YOLOv8 inference on the frame
                results = modelx(imgRegion2, stream=True, verbose=False)
                detections2 = np.empty((0, 5))

                # Visualize the results on the frame
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
                            currentArray2 = np.array([x1, y1, x2, y2, conf])
                            detections2 = np.vstack((detections2, currentArray2))

                resultsTracker2 = tracker2.update(detections2)

                if showParkingLotLine:
                    for line in [MiddleCamMiddleEntryLine, MiddleCamMiddleExitLine, MiddleCamBottomEntryLine,
                                 MiddleCamBottomExitLine]:
                        cv2.line(frame2, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)

                for results in resultsTracker2:
                    x1, y1, x2, y2, id = map(int, results)
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2

                    if showCarID:
                        cvzone.putTextRect(frame2, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                                           offset=10)
                    if showCarCenterDot:
                        cv2.circle(frame2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    if showCarRecBox:
                        cvzone.cornerRect(frame2, (x1, y1, w, h), l=9, rt=5)

                    update_car_info(id, x1, y1, x2, y2, car_dict, "MiddleCam")

                    if check_car_cross_line2(MiddleCamMiddleEntryLine, cx, cy):
                        cv2.line(frame2, (MiddleCamMiddleEntryLine[0], MiddleCamMiddleEntryLine[1]),
                                 (MiddleCamMiddleEntryLine[2], MiddleCamMiddleEntryLine[3]), (0, 255, 0), 5)
                        if id in car_temp4 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam":
                            car_temp4[id] = car_dict[id]
                            if car_dict[id]["carPlate"] == "-":
                                for results in resultsTracker:
                                    x1, y1, x2, y2, id2 = results
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    if id2 in car_temp3 and id2 in car_dict and car_dict[id2]["status"] == "moving" and \
                                            car_dict[id2]["currentLocation"] == "RightCam":
                                        car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                        car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                        del car_temp3[id2]
                        else:
                            car_temp4[id] = copy_car_dict.copy()
                            car_temp4[id]["bbox"] = (x1, y1, x2, y2)
                            car_temp4[id]["currentLocation"] = "MiddleCam"
                            if car_dict[id]["carPlate"] == "-":
                                for results in resultsTracker:
                                    x1, y1, x2, y2, id2 = results
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    if id2 in car_temp3 and id2 in car_dict and car_dict[id2]["status"] == "moving" and \
                                            car_dict[id2]["currentLocation"] == "RightCam":
                                        car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                        car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                        del car_temp3[id2]

                    if check_car_cross_line2(MiddleCamMiddleExitLine, cx, cy):
                        cv2.line(frame2, (MiddleCamMiddleExitLine[0], MiddleCamMiddleExitLine[1]),
                                 (MiddleCamMiddleExitLine[2], MiddleCamMiddleExitLine[3]), (0, 255, 0), 5)
                        if id in car_temp7 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam" and \
                                car_dict[id]["status"] == "moving":
                            car_temp7[id] = car_dict[id]
                        else:
                            car_temp7[id] = copy_car_dict.copy()
                            car_temp7[id]["bbox"] = (x1, y1, x2, y2)
                            car_temp7[id]["currentLocation"] = "MiddleCam"

                    if check_car_cross_line3(MiddleCamBottomEntryLine, cx, cy):
                        cv2.line(frame2, (MiddleCamBottomEntryLine[0], MiddleCamBottomEntryLine[1]),
                                 (MiddleCamBottomEntryLine[2], MiddleCamBottomEntryLine[3]), (0, 255, 0), 5)
                        if id in car_temp2 and id in car_dict and car_dict[id]["currentLocation"] == "MiddleCam":
                            car_temp2[id] = car_dict[id]
                            if car_dict[id].get("carPlate") == "-":
                                for results in resultsTracker:
                                    x1, y1, x2, y2, id2 = results
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    if id2 in car_temp and id2 in car_dict and car_dict[id2]["status"] == "moving" and \
                                            car_dict[id2]["currentLocation"] == "RightCam":
                                        car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                        car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                        car_dict[id2]["currentLocation"] = "MiddleCam"
                                        del car_temp[id2]
                        else:
                            car_temp2[id] = copy_car_dict.copy()
                            car_temp2[id]["bbox"] = (x1, y1, x2, y2)
                            car_temp2[id]["currentLocation"] = "MiddleCam"

                            if id in car_dict and "carPlate" in car_dict[id]:
                                if (car_dict[id]["carPlate"] == "-"):
                                    for results in resultsTracker:
                                        x1, y1, x2, y2, id2 = results
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if id2 in car_temp and id2 in car_dict and car_dict[id2][
                                            "status"] == "moving" and \
                                                car_dict[id2]["currentLocation"] == "RightCam":
                                            car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                            car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                            car_dict[id2]["currentLocation"] = "MiddleCam"
                                            del car_temp[id2]

                    if check_car_cross_line3(MiddleCamBottomExitLine, cx, cy):
                        cv2.line(frame2, (MiddleCamBottomExitLine[0], MiddleCamBottomExitLine[1]),
                                 (MiddleCamBottomExitLine[2], MiddleCamBottomExitLine[3]), (0, 255, 0), 5)

                    # Draw Parking Box coordinate
                    draw_parking_lot_polylines(frame2, ParkingLotCoordinateMidCam, showParkingLotBox, parking_lots,
                                               car_dict)
                    draw_parking_lot_polylines(frame2, DoubleParkCoordinateMidCam, showDoubleParkingLotBox,
                                               double_park_lots, car_dict)

                    # Run Parking Detection Algorithm in Middle Camera
                    for parking_lot_name in parking_lots_data_mid_cam.keys():
                        process_parking(parking_lots_data_mid_cam, car_dict, id, cx, cy, parking_lot_name, parking_lots)

                    for parking_lot_name in DoubleParkCoordinateMidCam.keys():
                        process_double_parking(DoubleParkCoordinateMidCam, car_dict, id, cx, cy, parking_lot_name,
                                               double_park_lots, parking_lots)

                # Read a frame from the video
                success3, frame3 = cap3.read()

                if success3 and frame3 is not None and stop_all_func is False:
                    # Start the timer
                    start3 = timeit.default_timer()

                    imgRegion3 = cv2.bitwise_and(frame3, mask3)

                    # Run YOLOv8 inference on the frame
                    results = modelm(imgRegion3, stream=True, verbose=False)
                    detections3 = np.empty((0, 5))

                    # Visualize the results on the frame
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
                                currentArray3 = np.array([x1, y1, x2, y2, conf])
                                detections3 = np.vstack((detections3, currentArray3))

                    resultsTracker3 = tracker3.update(detections3)

                    if showParkingLotLine:
                        for line in [LeftCamMiddleEntryLine, LeftCamMiddleExitLine, LeftCamBottomEntryLine,
                                     LeftCamBottomExitLine]:
                            cv2.line(frame3, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)

                    for results in resultsTracker3:
                        x1, y1, x2, y2, id = map(int, results)
                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w // 2, y1 + h // 2

                        if showCarID:
                            cvzone.putTextRect(frame3, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                                               offset=10)

                        if showCarCenterDot:
                            cv2.circle(frame3, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                        if showCarRecBox:
                            cvzone.cornerRect(frame3, (x1, y1, w, h), l=9, rt=5)

                        # Check if the car ID already exists in the dictionary
                        update_car_info(id, x1, y1, x2, y2, car_dict, "LeftCam")

                        if check_car_cross_line3(LeftCamMiddleEntryLine, cx, cy):
                            if id not in car_temp8:
                                cv2.line(frame3, (LeftCamMiddleEntryLine[0], LeftCamMiddleEntryLine[1]),
                                         (LeftCamMiddleEntryLine[2], LeftCamMiddleEntryLine[3]), (0, 255, 0), 5)
                            if id in car_temp8 and id in car_dict and car_dict[id]["currentLocation"] == "LeftCam":
                                car_temp8[id] = copy_car_dict.copy()
                                car_temp8[id]["bbox"] = (x1, y1, x2, y2)
                                car_temp8[id]["currentLocation"] = "LeftCam"
                                if car_dict[id]["carPlate"] == "-":
                                    for results in resultsTracker2:
                                        x1, y1, x2, y2, id2 = results
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if id2 in car_temp7 and id2 in car_dict and car_dict[id2][
                                            "status"] == "moving" and \
                                                car_dict[id2]["currentLocation"] == "MiddleCam":
                                            car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                            car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                            car_dict[id2]["currentLocation"] = "LeftCam"
                                            del car_temp7[id2]
                            else:
                                car_temp8[id] = copy_car_dict.copy()
                                car_temp8[id]["bbox"] = (x1, y1, x2, y2)
                                car_temp8[id]["currentLocation"] = "LeftCam"
                                if car_dict[id]["carPlate"] == "-":
                                    for results in resultsTracker2:
                                        x1, y1, x2, y2, id2 = results
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if id2 in car_temp7 and id2 in car_dict and car_dict[id2][
                                            "status"] == "moving" and \
                                                car_dict[id2]["currentLocation"] == "MiddleCam":
                                            car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                            car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                            car_dict[id2]["currentLocation"] = "LeftCam"
                                            del car_temp7[id2]

                        if check_car_cross_line3(LeftCamBottomEntryLine, cx, cy):
                            cv2.line(frame3, (LeftCamBottomEntryLine[0], LeftCamBottomEntryLine[1]),
                                     (LeftCamBottomEntryLine[2], LeftCamBottomEntryLine[3]), (0, 255, 0), 5)
                            if id in car_temp6 and id in car_dict and car_dict[id]["currentLocation"] == "LeftCam":
                                car_temp6[id] = car_dict[id]
                                if car_dict[id]["carPlate"] == "-":
                                    for results in resultsTracker2:
                                        x1, y1, x2, y2, id2 = results
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if id2 in car_temp5 and id2 in car_dict and car_dict[id2][
                                            "status"] == "moving" and \
                                                car_dict[id2]["currentLocation"] == "MiddleCam":
                                            car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                            car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                            car_dict[id2]["currentLocation"] = "LeftCam"
                                            del car_temp5[id2]
                            else:
                                car_temp6[id] = copy_car_dict.copy()
                                car_temp6[id]["bbox"] = (x1, y1, x2, y2)
                                car_temp6[id]["currentLocation"] = "LeftCam"
                                if car_dict[id]["carPlate"] == "-":
                                    for results in resultsTracker2:
                                        x1, y1, x2, y2, id2 = results
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if id2 in car_temp5 and id2 in car_dict and car_dict[id2][
                                            "status"] == "moving" and \
                                                car_dict[id2]["currentLocation"] == "MiddleCam":
                                            car_dict[id]["carPlate"] = car_dict[id2]["carPlate"]
                                            car_dict[id]["entryTime"] = car_dict[id2]["entryTime"]
                                            car_dict[id2]["currentLocation"] = "LeftCam"
                                            del car_temp5[id2]

                        if check_car_cross_line1(LeftCamBottomExitLine, cx, cy):
                            cv2.line(frame3, (LeftCamBottomExitLine[0], LeftCamBottomExitLine[1]),
                                     (LeftCamBottomExitLine[2], LeftCamBottomExitLine[3]), (0, 255, 0), 5)
                            if id in car_dict:
                                car_dict[id]["exitTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                        if check_car_cross_line3(LeftCamMiddleExitLine, cx, cy):
                            cv2.line(frame3, (LeftCamMiddleExitLine[0], LeftCamMiddleExitLine[1]),
                                     (LeftCamMiddleExitLine[2], LeftCamMiddleExitLine[3]), (0, 255, 0), 5)
                            if id in car_dict:
                                car_dict[id]["exitTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                        # draw Parking Box coordinate
                        draw_parking_lot_polylines(frame3, ParkingLotCoordinateLeftCam, showParkingLotBox, parking_lots,
                                                   car_dict)
                        draw_parking_lot_polylines(frame3, DoubleParkCoordinateLeftCam, showDoubleParkingLotBox,
                                                   double_park_lots, car_dict)

                        # Run Parking Detection Algorithm in Middle Camera
                        for parking_lot_name in parking_lots_data_left_cam.keys():
                            process_parking(parking_lots_data_left_cam, car_dict, id, cx, cy, parking_lot_name,
                                            parking_lots)

                        for parking_lot_name in DoubleParkCoordinateLeftCam.keys():
                            process_double_parking(DoubleParkCoordinateLeftCam, car_dict, id, cx, cy, parking_lot_name,

                                                   double_park_lots, parking_lots)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q") or stop_all_func:
                        stop_all_func = True
                        break
                else:
                    # Break the loop if the end of the video is reached
                    stop_all_func = True
                    break

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_all_func:
                    stop_all_func = True
                    break
            else:
                # Break the loop if the end of the video is reached
                stop_all_func = True
                break
            #print(car_dict)

            #To display Video Output
            ShowVideoOutput(frame1,frame2,frame3, start1, start2, start3, parking_lots)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_all_func:
                stop_all_func = True
                break
        else:
            # Break the loop if the end of the video is reached
            stop_all_func = True
            app.quit()
            break

    # Release video capture objects and close windows
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()


