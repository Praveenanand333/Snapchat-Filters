import tkinter as tk
import cv2
import datetime

def apply_filter(frame):
    # Apply Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return blurred_frame

def apply_blur():
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        if ret:
            # Apply filter
            filtered_frame = apply_filter(frame)

            # Display the filtered frame
            cv2.imshow('Filtered Feed', filtered_frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()

apply_blur()
