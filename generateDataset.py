#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:19:45 2025

@author: corner
"""

import cv2
from FaceDescriptor import FaceDescriptor
import os
import numpy as np
import pickle

descriptor = FaceDescriptor( withAlignment=True, alignmentType='proposed')

obj = os.scandir("dataset")
dataX = []
dataY = []
classIndex = []
for entry in obj:
    if entry.is_file():     
        print(entry.name)
        classIndex.append(entry.name)
        cap = cv2.VideoCapture(entry.path)         
       
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # cv2.imshow('Frame',frame)
                opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                face_img = descriptor.extractFacial(opencv_image)
                if face_img is not None:
                    feature = descriptor.extractFacialFeatures(face_img)
                    dataX.append(feature)
                    dataY.append(classIndex.index(entry.name))
            else: 
              break         
        cap.release()

with open('rt1_aligned_facial_features.pickle', 'wb') as handle:
    pickle.dump([dataX, dataY, classIndex], handle, protocol=pickle.HIGHEST_PROTOCOL)