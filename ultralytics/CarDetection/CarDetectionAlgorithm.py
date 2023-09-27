from datetime import datetime

import cv2
import numpy as np

from CarDetection.CarDetectController import PARKING_TIME, DOUBLE_PARK_TIME
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


def draw_parking_lot_polylines(frame, coordinates, show_boxes, parking_lots, car_dict):
    if show_boxes:
        for coord_name, coord_points in coordinates.items():
            parking_status = parking_lots.get(coord_name, {})
            is_double_parked = False

            # Check if the current parking lot has carId information
            car_ids = parking_status.get("carId")
            if car_ids is not None:
                if isinstance(car_ids, int):
                    car_ids = [car_ids]  # Convert a single int to a list
                for car_id in car_ids:
                    car_data = car_dict.get(car_id, {})
                    if car_data.get("isDoubleParked") == "yes":
                        is_double_parked = True
                        break

            if is_double_parked:
                color = (255, 0, 0)  # Blue for double-parked cars
            elif parking_status.get("parked") == "no":
                color = (0, 255, 0)  # Green for empty parking
            else:
                color = (0, 0, 255)  # Red for occupied parking

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

            if duration > PARKING_TIME:
                car_data["status"] = "parked"
                parking_lots[parking_lot_name]["carId"] = id
                parking_lots[parking_lot_name]["parked"] = "yes"
            else:
                car_data["status"] = "moving"

        else:
            # Car entered the parking lot
            car_data["parkingDetected"] = current_time
            car_data["parkingLot"] = parking_lot_name
            parking_lots[parking_lot_name]["carId"] = id
            parking_lots[parking_lot_name]["parked"] = "no"
            car_data["status"] = "moving"
            car_data["exitTime"] = "-"

        car_dict[id] = car_data

    else:
        # Car is no longer inside the parking lot
        car_data = car_dict.get(id, {})
        if car_data.get("parkingLot") == parking_lot_name:
            parking_lots[parking_lot_name]["carId"] = "-"
            parking_lots[parking_lot_name]["parked"] = "no"
            car_data["parkingDetected"] = "-"
            car_data["parkingLot"] = "-"
            car_data["status"] = "moving"
            car_data["duration"] = "-"
            car_dict[id] = car_data


def process_double_parking(coordinates, car_dict, id, cx, cy, parking_lot_name, double_parking_lots, parking_lots):
    parking_lot_coordinates = np.array(coordinates.get(parking_lot_name, []), np.int32)
    if parking_lot_coordinates.size == 0:
        return  # Invalid parking lot name

    results = cv2.pointPolygonTest(parking_lot_coordinates, (cx, cy), False)

    # Initialize the list to keep track of double park cars
    double_parked_cars = []

    if results >= 0:
        # Car is inside the parking lot
        car_data = car_dict.get(id, {})
        current_time = datetime.now()

        if car_data.get("parkingLot") == parking_lot_name:
            duration = (current_time - car_data.get("parkingDetected", current_time)).seconds
            car_data["duration"] = duration

            if duration > DOUBLE_PARK_TIME:
                car_data["status"] = "parked"
                double_parking_lots[parking_lot_name]["carId"] = id
                double_parking_lots[parking_lot_name]["parked"] = "yes"
                car_data["isDoubleParking"] = "yes"

                # Set isDoubleParked to "yes" for other cars that are parked in covered_parking_lots
                all_covered_parking_lot = double_parking_lots[parking_lot_name]["covered_parking_lot"]
                for covered_parking_lot in all_covered_parking_lot:
                    if covered_parking_lot in parking_lots and parking_lots[covered_parking_lot]["parked"] == "yes":
                        car_id = parking_lots[covered_parking_lot].get("carId", "-")
                        if car_id != "-" and car_id != id:
                            double_parked_cars.append(car_id)

                # Update the double parked cars' data
                for double_parked_car_id in double_parked_cars:
                    if car_dict[double_parked_car_id]["isDoubleParked"] == "no":
                        car_dict[double_parked_car_id]["isDoubleParked"] = "yes"
                        car_dict[double_parked_car_id]["doubleParkingLot"] = parking_lot_name

            else:
                car_data["status"] = "moving"

        else:
            # Car entered the parking lot
            car_data["parkingDetected"] = current_time
            car_data["parkingLot"] = parking_lot_name
            double_parking_lots[parking_lot_name]["carId"] = id
            double_parking_lots[parking_lot_name]["parked"] = "no"
            car_data["status"] = "moving"
            car_data["exitTime"] = "-"

        car_dict[id] = car_data

    else:
        # Car is no longer inside the parking lot
        car_data = car_dict.get(id, {})
        if car_data.get("parkingLot") == parking_lot_name:
            car_data["parkingDetected"] = "-"
            car_data["parkingLot"] = "-"
            car_data["status"] = "moving"
            car_data["duration"] = "-"
            car_data["isDoubleParking"] = "no"

            if double_parking_lots[parking_lot_name]["carId"] == id:
                # Remove the car ID from double_parking_lots
                double_parking_lots[parking_lot_name]["carId"] = "-"

                if double_parking_lots[parking_lot_name]["parked"] == "yes":
                    # Set isDoubleParked to "yes" for other cars that are parked in covered_parking_lots
                    all_covered_parking_lot = double_parking_lots[parking_lot_name]["covered_parking_lot"]
                    for covered_parking_lot in all_covered_parking_lot:
                        if covered_parking_lot in parking_lots and parking_lots[covered_parking_lot]["parked"] == "yes":
                            car_id = parking_lots[covered_parking_lot].get("carId", "-")
                            if car_id != "-" and car_id != id:
                                double_parked_cars.append(car_id)

                    # Clear isDoubleParked and doubleParkingLot for any blocked cars
                    for double_parked_car_id in double_parked_cars:
                        car_dict[double_parked_car_id]["doubleParkingLot"] = "-"
                        car_dict[double_parked_car_id]["isDoubleParked"] = "no"

                double_parking_lots[parking_lot_name]["parked"] = "no"

    return car_dict, double_parking_lots



