#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 08:44:28 2025

@author: corner
"""



import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import math
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization



class FaceDescriptor():
    def __init__(self, withAlignment = False, alignmentType = 'proposed'):

        self.alignmentType = alignmentType
        self.withAlignment = withAlignment
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Create a face detector instance with the image mode:

        
        if withAlignment:
            base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
            
            options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                    output_face_blendshapes=False,
                                                    output_facial_transformation_matrixes=False,                                     
                                                    num_faces=1)        
            self.detector = vision.FaceLandmarker.create_from_options(options)
            
        else:
            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
                running_mode=VisionRunningMode.IMAGE)
            self.detector = FaceDetector.create_from_options(options)
        
        
        
        self.extractor = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
            transforms.Resize((160,160),antialias=True),    
            transforms.ConvertImageDtype(torch.float32)
            ])
        
        
      
        
    def extractFacialFeatures(self, face_img):
        inp = self.transform(face_img)
        with torch.no_grad():
            features = self.extractor(inp.unsqueeze(0)).data[0]
        return features.cpu().numpy()


    def getAngle(self, left_eye, right_eye):
        
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye
        
        delta =  left_eye - right_eye
        if delta[1] > 0:
            t = -1
        else:
            t = 1
        delta = np.abs(delta)
   
        return  t *  np.arctan(delta[1]/delta[0]) * 180 / math.pi
        
    
    
    def getRotatedLandmark(self, lanmarks, angle, width, height):
        theta = (angle  * math.pi) / 180
        
        cx = width / 2
        cy = height / 2
        
        
        rotated_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        #xyxy
        box = [height, width, 0, 0]
        for v in lanmarks:
            x = v.x * width
            y = v.y * height
            tempX = x - cx
            tempY = y - cy
            
            rotatedX = tempX*math.cos(theta) - tempY*math.sin(theta)
            rotatedY = tempX*math.sin(theta) + tempY*math.cos(theta)
            
           
            x = rotatedX + cx
            y = rotatedY + cy
            
            rotated_landmarks.landmark.append(
                landmark_pb2.NormalizedLandmark(x=x/width, y=y/height, z=v.z)
            )
            box[0] = int(max(min(box[0], x),0))
            box[1] = int(max(min(box[1], y),0))
            box[2] = int(min(max(box[2], x),width))
            box[3] = int(min(max(box[3], y),height))
          
        return rotated_landmarks, box
    
    def rotate_image(self, image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    
    def extractFacial(self, frame):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        new_img = np.copy(frame)   
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=new_img)
        detection_result = self.detector.detect(mp_image)
        
        if self.withAlignment:
            if len(detection_result.face_landmarks) > 0:
                # Step 1: find the original box
                landmarks = detection_result.face_landmarks[0]
                ori_box = [frame_width, frame_height,0,0]
                for v in landmarks:
                    ori_box[0] = max(min(ori_box[0], int(v.x * frame_width)),0)
                    ori_box[1] = max(min(ori_box[1], int(v.y * frame_height)),0)
                    ori_box[2] = min(max(ori_box[2], int(v.x * frame_width)),frame_width)
                    ori_box[3] = min(max(ori_box[3], int(v.y * frame_height)),frame_height)
                    
                # Step 2: Calculate the eye 
                left_eye = np.asarray([0.0,0.0])
                right_eye = np.asarray([0.0,0.0])
                for i in [474,475,476,477]:
                    left_eye[0] += landmarks[i].x * frame_width / 4
                    left_eye[1] += landmarks[i].y * frame_height / 4
                for i in [469,470,471,472]:
                    right_eye[0] += landmarks[i].x * frame_width / 4
                    right_eye[1] += landmarks[i].y * frame_height / 4                    
                angle = self.getAngle(left_eye, right_eye)     
                
                # Step 3: Rotate all the landmarks based on the eye angle
                landmarks, box = self.getRotatedLandmark(landmarks, angle,frame_width, frame_height)
                
                
                # Step 4: Calculate the offset based on the dataset settings
                offsetH = abs(192 - (box[3] - box[1]))
                offsetW = abs(169 - (box[2] - box[0]))
                
                if offsetH > offsetW:
                    ratio = 192 / (box[3] - box[1])
                else:
                    ratio = 169 /  (box[2] - box[0])
                
                if self.alignmentType == 'proposed':
                    # Step 5: Rotate the image and resize the image based on the ratio
                    new_img = self.rotate_image(new_img, -angle)
                    new_img = cv2.resize(new_img, (0, 0), fx = ratio, fy = ratio)
                    
                    # Step 6: Make the images to 200 px
                    scale = np.flipud(np.divide(new_img.shape[:2], frame.shape[:2])) 
                    box[:2] = np.multiply(box[:2], ratio )
                    box[2:] = np.multiply(box[2:], ratio )
                    offsetX = (200 - (box[2] - box[0]))/2
                    offsetY = (200 - (box[3] - box[1]))/2
                    
                    missingX = max((int(max(box[0] - offsetX,0))+ 200 ) - new_img.shape[1], 0)
                    missingY = max((int(max(box[1] - offsetY,0))+ 200 ) - new_img.shape[0], 0)
                    
                    
                    box[0] = int(max(box[0] - offsetX - missingX,0))
                    box[1] = int(max(box[1] - offsetY - missingY,0))
                    box[2] = box[0] + 200
                    box[3] = box[1] + 200
                    box = np.asarray(box, int)
                    
                    # Step 7: Crop the image
                    new_img = new_img[box[1]:box[3],box[0]:box[2]]
                
                elif self.alignmentType == 'alignThenCrop':
                    new_img = self.rotate_image(new_img, -angle)
                    new_img = new_img[box[1]:box[3],box[0]:box[2]]
                elif self.alignmentType == 'cropThenAlign':
                    new_img = frame[ori_box[1]:ori_box[3],ori_box[0]:ori_box[2]]
                    new_img = self.rotate_image(new_img, -angle)
                    
                elif self.alignmentType == 'noAlign':
                    new_img = frame[ori_box[1]:ori_box[3],ori_box[0]:ori_box[2]]
                
                return new_img
                
                
                
        else: # no alignment, direct return imgs
            if len(detection_result.detections) == 0:
                return None
            box = detection_result.detections[0].bounding_box
            xyxy = [box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height]
            new_img = new_img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]]            
            return new_img
    

# Uncomment below for testing purpose.
# # Create the descriptor
# descriptor = FaceDescriptor( withAlignment=True, alignmentType='proposed')

# # Read the image
# img = cv2.imread('material/test_face.jpg')
# # must convert to RGB arrangement
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# # Extract the facial image from the image
# result = descriptor.extractFacial(img)
# # Display it
# result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) 
# cv2.imshow('img', result) 
# cv2.waitKey(0)   
# cv2.destroyAllWindows()
# # Extract the facial feature
# feature = descriptor.extractFacialFeatures(result)
# print(feature)

