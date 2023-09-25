from datetime import datetime

import cv2
import numpy as np

from CarDetection.CarDetectController import PARKING_TIMEOUT
from CarDetection.config import copy_car_dict

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def update_car_info(id, x1, y1, x2, y2, car_dict, current_location):
    if id in car_dict:
        # Update the bounding box coordinates for the existing car ID
        car_dict[id]["bbox"] = (x1, y1, x2, y2)
    else:
        # Add a new entry to the dictionary for the new car ID
        car_dict[id] = copy_car_dict.copy()
        car_dict[id]["bbox"] = (x1, y1, x2, y2)
        car_dict[id]["entryTime"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        car_dict[id]["currentLocation"] = current_location

def check_car_cross_line1(line, cx, cy):
    x1, y1, x2, y2 = line
    result = x1 < cx < x2 and y1 > cy > y2
    return result

def check_car_cross_line2(line, cx, cy):
    x1, y1, x2, y2 = line
    result = x1 < cx < x2 and y1 < cy < y2
    return result

def check_car_cross_line3(line, cx, cy):
    x1, y1, x2, y2 = line
    result = x1 > cx > x2 and y1 < cy < y2
    return result


def draw_parking_lot_polylines(frame, coordinates, show_boxes, parking_lots):
    if(show_boxes):
        for coord_name, coord_points in coordinates.items():
            parking_status = parking_lots.get(coord_name, {"carId": "unknown", "parked": "no"})
            color = (0, 255, 0) if parking_status["parked"] == "no" else (0, 0, 255)
            cv2.polylines(frame, [np.array(coord_points, np.int32)], True, color, 2)

def process_parking(coordinates, car_dict, id, cx, cy, parking_lot_name, parking_lots):
    parking_lot_coordinates = np.array(coordinates.get(parking_lot_name, []), np.int32)
    if parking_lot_coordinates.size == 0:
        return  # Invalid parking lot name

    results = cv2.pointPolygonTest(parking_lot_coordinates, (cx, cy), False)

    if results >= 0:
        # Car is inside the parking lot
        car_data = car_dict.get(id, {})
        current_time = datetime.now()

        if car_data.get("parkingLot") == parking_lot_name:
            duration = (current_time - car_data.get("parkingDetected", current_time)).seconds
            car_data["duration"] = duration

            if duration > PARKING_TIMEOUT:
                car_data["status"] = "parked"
                parking_lots[parking_lot_name]["carId"] = id
                parking_lots[parking_lot_name]["parked"] = "yes"
            else:
                car_data["status"] = "moving"
                parking_lots[parking_lot_name]["carId"] = "unknown"
                parking_lots[parking_lot_name]["parked"] = "no"
        else:
            # Car entered the parking lot
            car_data["parkingDetected"] = current_time
            car_data["parkingLot"] = parking_lot_name
            parking_lots[parking_lot_name]["carId"] = id
            parking_lots[parking_lot_name]["parked"] = "no"
            car_data["status"] = "moving"
            car_data["exitTime"] = "unknown"

        car_dict[id] = car_data

    else:
        # Car is no longer inside the parking lot
        car_data = car_dict.get(id, {})
        if car_data.get("parkingLot") == parking_lot_name:
            parking_lots[parking_lot_name]["carId"] = "unknown"
            parking_lots[parking_lot_name]["parked"] = "no"
            car_data["parkingDetected"] = "unknown"
            car_data["parkingLot"] = "unknown"
            car_data["status"] = "moving"
            car_data["duration"] = "unknown"
            car_dict[id] = car_data