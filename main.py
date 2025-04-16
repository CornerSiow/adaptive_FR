#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:18:42 2025

@author: corner
"""


import cv2
import numpy as np
from FaceDscriptor import FaceDescriptor
import os
import time
import pickle
import random
from GNG import GNG
from sklearn.metrics.pairwise import cosine_similarity

from SiameseNetworkLearning import SiameseModel

# Set the threshold
threshold = 0.8

# a function to find the maximum similarity from the anchor image
def getMaxSimilarity(inp, data):
    maxSimilarity = 0
    for f in data:
        d = cosine_similarity([inp], [f])
        maxSimilarity = max(d[0][0], maxSimilarity)    
    return maxSimilarity

# Load the extracted facial database.
with open('material/rt1_aligned_facial_features.pickle', 'rb') as handle:
    dataX, dataY, className = pickle.load(handle)
    
# Obtains the data Y index.
allClassIndexes = np.unique(dataY)

# obtains the training data for each person. 
# In this case, we get the first 20% frames for training.
trainDataX = []
trainDataY = []
for classIndex in allClassIndexes:
    tempDataX = dataX[dataY == classIndex]
    tempDataY = dataY[dataY == classIndex]
    trainDataX.extend(tempDataX[:len(tempDataX)//5])
    trainDataY.extend(tempDataY[:len(tempDataY)//5])
trainDataX = np.asarray(trainDataX)
trainDataY = np.asarray(trainDataY)


# Create the siamese network using CPU device (can use Cuda if available)
model = SiameseModel('cpu')
# Train the siamese model with 10 epochs
model.train(trainDataX, trainDataY, epoch=10)


# Now start testing
# Set the face descriptor to detect the face from the video.
descriptor = FaceDescriptor(withAlignment=True, alignmentType='proposed')       


# Create the GNG model for short-term memory system
gng = GNG(512)
anchorImage = []
# Read the video.
cap = cv2.VideoCapture('material/corner.m4v')    
while(cap.isOpened()):
    ret, frame = cap.read()
    displayStr= "Face Not Detected"
    if ret == True:              
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        face = descriptor.extractFacial(opencv_image)
                    
        similarityResult = 0
        if face is not None:
            feature = descriptor.extractFacialFeatures(face)
            
            # Use the trained Siamese network to ouput the latent feature.
            # Can comment it to see the effect.
            feature = model.process(feature)
            
            # Store the first 5 frames as anchor images
            if len(anchorImage) < 5:
                anchorImage.append(feature)
                if len(anchorImage) == 2:
                    gng.initializing(anchorImage[0], anchorImage[1])
                elif len(anchorImage) > 2:
                    gng.pushData(feature)    
                similarityResult = 1
            else: # now start evaluating the recognition
                # first compare with anchor image
                similarityResult = getMaxSimilarity(feature, anchorImage)
                if similarityResult >= threshold:
                    gng.pushData(feature)
                # If failed, then compare with GNG nodes.
                else:
                    similarityResult = getMaxSimilarity(feature, gng.nodeList[:,:-1])
                    if similarityResult >= threshold:
                        gng.pushData(feature)    
               
            if similarityResult > threshold:
                displayStr = "Authorized Person : {:.2f}".format(similarityResult)
            else:
                displayStr = "Unauthorized Person : {:.2f}".format(similarityResult)
         
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    
        frame = cv2.putText(frame, displayStr , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        cv2.waitKey(1)
      
    else: 
      break         
cap.release()
