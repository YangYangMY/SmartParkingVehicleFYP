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

# Define the reference video stream
reference_video = "CarDetection/VideoFootage/rightCamera1080.mp4"  # Choose one as the reference

# Calculate the reference video's frame rate
reference_cap = cv2.VideoCapture(reference_video)
reference_frame_rate = int(reference_cap.get(cv2.CAP_PROP_FPS))
reference_cap.release()

def synchronize_videos(video_cap, timestamp, reference_frame_time, window_name):
    while True:
        ret, frame = video_cap.read()
        if not ret or frame is None:
            break

        # Calculate the current frame's timestamp
        current_timestamp = cv2.getTickCount()
        elapsed_time = (current_timestamp - timestamp) / cv2.getTickFrequency()

        # Calculate the target frame time based on reference frame rate
        target_time = reference_frame_time + (elapsed_time * reference_frame_rate)

        # Wait until the target time is reached
        while (cv2.getTickCount() / cv2.getTickFrequency()) < target_time:
            pass

        # Display the frame with a timestamp
        timestamp_text = f"Timestamp: {target_time:.2f} seconds"
        cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


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
        export_thread = threading.Thread(target=export_data_to_excel_thread, args=(ExportExcel_filename, formatted_current_time, car_dict_column_names, car_dict, parking_lots_column_names,  parking_lots, double_park_lots_column_names, double_park_lots))

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

        print("Start ALL Camera THREAD")
        # Start the threads
        thread_camera1.start()
        thread_camera2.start()
        thread_camera3.start()

        print("starting license plate thread")
        process_car_plate_video_thread.start()


        print("starting export thread")
        export_thread.start()

        # Open the reference video
        reference_cap = cv2.VideoCapture(reference_video)

        # Synchronize the other videos to the reference video's timing
        while True:
            ret, _ = reference_cap.read()
            if not ret:
                break

            reference_timestamp = cv2.getTickCount()
            synchronize_videos(cv2.VideoCapture("CarDetection/VideoFootage/middleCamera1080.mp4"), reference_timestamp, reference_timestamp, "Camera 2")
            synchronize_videos(cv2.VideoCapture("CarDetection/VideoFootage/leftCamera1080.mp4"), reference_timestamp, reference_timestamp, "Camera 3")
            synchronize_videos(cv2.VideoCapture("LicensePlateRecognition/entryCamera1080.mp4"), reference_timestamp, reference_timestamp, "OCR Detection")


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