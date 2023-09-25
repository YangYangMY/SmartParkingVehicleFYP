from config import *

PARKING_TIMEOUT = 30  # seconds


def draw_parking_lot_polylines(frame, coordinates, show_boxes):
    if show_boxes:
        for coord_name, coord_points in coordinates.items():
            cv2.polylines(frame, [np.array(coord_points, np.int32)], True, (0, 0, 255), 2)

def process_parking(coordinates, car_dict, id, cx, cy, parking_lot_name):
    results = cv2.pointPolygonTest(np.array(coordinates[parking_lot_name], np.int32), (cx, cy), False)
    if results >= 0:
        if car_dict[id]["parkingLot"] == parking_lot_name:
            car_dict[id]["duration"] = (datetime.now() - car_dict[id]["parkingDetected"]).seconds
            if car_dict[id]["duration"] > 30:
                car_dict[id]["status"] = "parked"
                if car_dict[id]["status"] == "parked":
                    ParkingLot[parking_lot_name]["parked"] = "yes"
                    ParkingLot[parking_lot_name]["carId"] = id
                else:
                    ParkingLot[parking_lot_name]["parked"] = "no"
                    ParkingLot[parking_lot_name]["carId"] = "unknown"
            else:
                car_dict[id]["status"] = "moving"
        else:
            car_dict[id]["parkingDetected"] = datetime.now()
            car_dict[id]["parkingLot"] = parking_lot_name
            ParkingLot[parking_lot_name] = {"carId": "unknown", "parked": "no"}
            car_dict[id]["status"] = "moving"
