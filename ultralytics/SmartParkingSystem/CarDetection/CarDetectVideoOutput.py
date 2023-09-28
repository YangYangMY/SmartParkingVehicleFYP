import time
import timeit

import cv2

from SmartParkingSystem.CarDetection.CarDetectController import showResizedVideo, showFPS, showParkingOccupancy


def ShowVideoOutput(frame, start_time, window_name, parking_lots):
    if (showResizedVideo):
        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame = cv2.resize(frame, (800, 600))

        if (showFPS):
            # Stop the timer and calculate the FPS
            end = time.time()
            yolo_fps = 1 / (end - start_time)

            # Display the frame rate on the left top of the screen
            cv2.putText(resized_frame, "FPS: " + "{:.2f}".format(yolo_fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        # Calculate parking occupancy
        if showParkingOccupancy:
            total_parking_spaces = len(parking_lots)
            occupied_spaces = sum(1 for space_info in parking_lots.values() if space_info['parked'] == 'yes')

            if total_parking_spaces > 0:
                parkingOccupancy = total_parking_spaces - occupied_spaces
            else:
                parkingOccupancy = 0

            cv2.putText(resized_frame, "Available Spaces: " + str(parkingOccupancy), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, resized_frame)

    else:
        if (showFPS):
            # Stop the timer and calculate the FPS
            end = timeit.default_timer()
            yolo_fps = 1 / (end - start_time)

            # Display the frame rate on the left top of the screen
            cv2.putText(frame, "FPS: " + "{:.2f}".format(yolo_fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        # Calculate parking occupancy
        if showParkingOccupancy:
            total_parking_spaces = len(parking_lots)
            occupied_spaces = sum(1 for space_info in parking_lots.values() if space_info['parked'] == 'yes')

            if total_parking_spaces > 0:
                parkingOccupancy = total_parking_spaces - occupied_spaces
            else:
                parkingOccupancy = 0

            cv2.putText(frame, "Available Spaces: " + str(parkingOccupancy), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        # cv2.setMouseCallback(window_name, RGB)