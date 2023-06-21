import tkinter as tk
import cv2

def apply_filter(frame):
    # Apply oil paint effect to the frame
    oil_painted_frame = cv2.xphoto.oilPainting(frame, 7, 1)

    return oil_painted_frame

def apply_oilpaint():
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

apply_oilpaint()
