# Main function
import logging
import os
import threading
from datetime import datetime

import cv2
import torch

from SmartParkingSystem.CarDetection.CarDetect import export_data_to_excel_thread, process_video_camera1, \
    process_video_camera2, process_video_camera3
from SmartParkingSystem.CarDetection.CarDetectController import ExportExcel_filename, car_dict_column_names, \
    parking_lots_column_names, double_park_lots_column_names
from SmartParkingSystem.CarDetection.config import car_dict, parking_lots, double_park_lots
from SmartParkingSystem.LicensePlateRecognition.LicensePlateRecognition import car_plate_video

def main():
    try:
        if torch.cuda.is_available():
            print("Running on GPU")
        else:
            print("Running on CPU")

        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print("Current time:", current_time)

        # Car Detection
        formatted_current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        export_thread = threading.Thread(target=export_data_to_excel_thread, args=(ExportExcel_filename, formatted_current_time, car_dict_column_names,
                                                                                   car_dict, parking_lots_column_names,  parking_lots, double_park_lots_column_names, double_park_lots))

        # Create a folder for log files (if it doesn't exist)
        log_folder = "logs"  # You can change this folder name to your preference
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Specify the path to the log file
        log_file_path = os.path.join(log_folder, f"{formatted_current_time}_error.log")

        # Configure logging settings
        logging.basicConfig(
            filename=log_file_path,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        )

        # License Plate Recognition
        process_car_plate_video_thread = threading.Thread(target=car_plate_video)

        # Create threads for each camera's video processing
        thread_camera1 = threading.Thread(target=process_video_camera1)
        thread_camera2 = threading.Thread(target=process_video_camera2)
        thread_camera3 = threading.Thread(target=process_video_camera3)

        # Start the threads
        thread_camera1.start()
        thread_camera2.start()
        thread_camera3.start()
        process_car_plate_video_thread.start()
        export_thread.start()

        # Wait for all threads to finish
        export_thread.join()
        process_car_plate_video_thread.join()
        thread_camera1.join()
        thread_camera2.join()
        thread_camera3.join()

        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")

if __name__ == "__main__":
    main()
