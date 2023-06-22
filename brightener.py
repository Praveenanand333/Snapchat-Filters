import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import tkinter as tk
import datetime

def adjust_brightness(frame, brightness):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Split the channels
    h, s, v = cv2.split(hsv)
    
    # Adjust the value (brightness) channel
    v = cv2.add(v, brightness)
    
    # Limit the pixel values to the valid range [0, 255]
    #v = cv2.clip(v, 0, 255)
    
    # Merge the channels back together
    hsv = cv2.merge((h, s, v))
    
    # Convert the frame back to the BGR color space
    adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_frame

def apply_brighten():
    
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        if ret:
            # Apply filter
            filtered_frame = adjust_brightness(frame,50)

            # Display the filtered frame
            cv2.imshow('BRIGHTENED FEED', filtered_frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Generate a unique filename based on the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"

            # Save the frame as an image
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as {filename}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()




   

