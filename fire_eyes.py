import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import datetime
mp_face_detection = mp.solutions.face_detection


face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
sample_img = cv2.imread('media/sample.jpg')

mp_face_mesh = mp.solutions.face_mesh


face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)


face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                         min_detection_confidence=0.5,min_tracking_confidence=0.3)

mp_drawing_styles = mp.solutions.drawing_styles
def detectFacialLandmarks(image, face_mesh, display = True):
    
    
    results = face_mesh.process(image[:,:,::-1])
    

    output_image = image[:,:,::-1].copy()
    if results.multi_face_landmarks:


        for face_landmarks in results.multi_face_landmarks:

    
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None, 
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

         
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None, 
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

   
    if display:
        

        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image);plt.title("Output");plt.axis('off');

    else:
        
        
        return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8), results              


def getSize(image, face_landmarks, INDEXES):
   
    image_height, image_width, _ = image.shape
    

    INDEXES_LIST = list(itertools.chain(*INDEXES))

    landmarks = []
 
    for INDEX in INDEXES_LIST:
        
    
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                               int(face_landmarks.landmark[INDEX].y * image_height)])
    

    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    
    landmarks = np.array(landmarks)
    
  
    return width, height, landmarks
def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
     
    
    
   
    image_height, image_width, _ = image.shape
    
    
    output_image = image.copy()
    
  
    status={}
    
   
    if face_part == 'MOUTH':
        
        
        INDEXES = mp_face_mesh.FACEMESH_LIPS
        
       
        loc = (10, image_height - image_height//40)
        
         
        increment=-30
        
        
    elif face_part == 'LEFT EYE':
        
        
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE
        
        
        loc = (10, 30)
        
    
        increment=30
    
   
    elif face_part == 'RIGHT EYE':
        
       
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE 
        
        
        loc = (image_width-300, 30)
        
        
        increment=30
    
    
    else:
        return
    
   
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        
         
        _, height, _ = getSize(image, face_landmarks, INDEXES)
        
         
        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        
        
        if (height/face_height)*100 > threshold:
            
            
            status[face_no] = 'OPEN'
            
            
            color=(0,255,0)
        
        
        else:
            
            status[face_no] = 'CLOSE'
            
            
            color=(0,0,255)
       
        cv2.putText(output_image, f'FACE {face_no+1} {face_part} {status[face_no]}.', 
                    (loc[0],loc[1]+(face_no*increment)), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)
                

    if display:

        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    
    else:
         
        return output_image, status
def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    
    
    
    annotated_image = image.copy()
    
   
    try:
    
       
        filter_img_height, filter_img_width, _  = filter_img.shape

       
        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)
        
       
        required_height = int(face_part_height*2.5)
        
         
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*
                                                         (required_height/filter_img_height)),
                                                     required_height))
        
      
        filter_img_height, filter_img_width, _  = resized_filter_img.shape

      
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        
        center = landmarks.mean(axis=0).astype("int")

       
        if face_part == 'MOUTH':

             
            location = (int(center[0] - filter_img_width / 3), int(center[1]))

       
        else:

            
            location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))

       
        ROI = image[location[1]: location[1] + filter_img_height,
                    location[0]: location[0] + filter_img_width]

     
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
        resultant_image = cv2.add(resultant_image, resized_filter_img)

        
        annotated_image[location[1]: location[1] + filter_img_height,
                        location[0]: location[0] + filter_img_width] = resultant_image
            
    
    except Exception as e:
        pass
    if display:

        plt.figure(figsize=[10,10])
        plt.imshow(annotated_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    
    else:
            
        return annotated_image
def apply_fire_eyes():
    import tkinter as tk
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

    cv2.namedWindow('Face Filter', cv2.WINDOW_NORMAL)

    left_eye = cv2.imread('media/left_eye.png')
    right_eye = cv2.imread('media/right_eye.png')

    smoke_animation = cv2.VideoCapture('media/smoke_animation.mp4')

    smoke_frame_counter = 0

    while camera_video.isOpened():
        
        ok, frame = camera_video.read()
        
    
        if not ok:
            continue
            
        
        _, smoke_frame = smoke_animation.read()
        
        
        smoke_frame_counter += 1
        
    
        if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
            
            smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        
            smoke_frame_counter = 0
        
        
        frame = cv2.flip(frame, 1)
        
    
        _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
        
        
        if face_mesh_results.multi_face_landmarks:
            
            
            _, mouth_status = isOpen(frame, face_mesh_results, 'MOUTH', 
                                        threshold=15, display=False)
            
            
            _, left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                            threshold=4.5 , display=False)
            
            
            _, right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                            threshold=4.5, display=False)
            
            
            for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                
                
                if left_eye_status[face_num] == 'OPEN':
                    
                
                    frame = overlay(frame, left_eye, face_landmarks,
                                    'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
                
            
                if right_eye_status[face_num] == 'OPEN':
                    

                    frame = overlay(frame, right_eye, face_landmarks,
                                    'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
                
            
                if mouth_status[face_num] == 'OPEN':
                    
                    
                    frame = overlay(frame, smoke_frame, face_landmarks, 
                                    'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
        
        cv2.imshow('Face Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Generate a unique filename based on the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"

            # Save the frame as an image
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as {filename}")

        k = cv2.waitKey(1) & 0xFF    
        
        if(k == 113):
            break
                    
    camera_video.release()
    cv2.destroyAllWindows()
