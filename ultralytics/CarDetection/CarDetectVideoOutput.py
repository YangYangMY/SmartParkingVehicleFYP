from CarDetectController import *
from CarDetection.CarDetectionAlgorithm import RGB
from config import *

def ShowVideoOutput(frame1, frame2, frame3, start1, start2, start3):
    if(showResizedVideo):
        # Resize the annotated frame to a maximum width and height of 600 pixels
        resized_frame1 = cv2.resize(frame1, (800, 600))
        resized_frame2 = cv2.resize(frame2, (800, 600))
        resized_frame3 = cv2.resize(frame3, (800, 600))

        if (showFPS):
            # Stop the timer and calculate the FPS
            end = timeit.default_timer()
            yolo_fps1 = 1 / (end - start1)
            yolo_fps2 = 1 / (end - start2)
            yolo_fps3 = 1 / (end - start3)

            # Display the frame rate on the left top of the screen
            cv2.putText(resized_frame1, "FPS: " + "{:.2f}".format(yolo_fps1), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)
            cv2.putText(resized_frame2, "FPS: " + "{:.2f}".format(yolo_fps2), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(resized_frame3, "FPS: " + "{:.2f}".format(yolo_fps3), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)

        if (showCombinedVideo):
            # Combine the frames side by side
            combined_frame = cv2.hconcat([resized_frame2, resized_frame1])

            # Display Combine frame
            cv2.imshow("Combined Video", combined_frame)
            cv2.imshow("Video 3", resized_frame3)
        else:
            if (showRightCam):
                cv2.imshow("Video 1", resized_frame1)
            if (showMidCam):
                cv2.imshow("Video 2", resized_frame2)
            if (showLeftCam):
                cv2.imshow("Video 3", resized_frame3)

    else:
        if (showFPS):
            # Stop the timer and calculate the FPS
            end = timeit.default_timer()
            yolo_fps1 = 1 / (end - start1)
            yolo_fps2 = 1 / (end - start2)
            yolo_fps3 = 1 / (end - start3)

            # Display the frame rate on the left top of the screen
            cv2.putText(frame1, "FPS: " + "{:.2f}".format(yolo_fps1), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)
            cv2.putText(frame2, "FPS: " + "{:.2f}".format(yolo_fps2), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame3, "FPS: " + "{:.2f}".format(yolo_fps3), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)

        if (showCombinedVideo):
            # Combine the frames side by side
            combined_frame = cv2.hconcat([frame2, frame1])

            # Display Combine frame
            cv2.imshow("Combined Video", combined_frame)
            cv2.imshow("Video 3", frame3)
        else:
            if (showRightCam):
                cv2.imshow("Video 1", frame1)
            if (showMidCam):
                cv2.imshow("Video 2", frame2)
            if (showLeftCam):
                cv2.imshow("Video 3", frame3)
                cv2.setMouseCallback('Video 3', RGB)




