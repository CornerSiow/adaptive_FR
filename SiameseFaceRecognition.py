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
from tqdm import tqdm

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from GNG import GNG
from sklearn.metrics.pairwise import cosine_similarity


# Siamese network.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            )

    def forward(self, x1):        
        x1 = self.fc(x1)       
        return x1

# constructive Loss function.
def constructiveLoss( x1, x2, Y):
    # Make sure the target minimum is value 1
    tempY = torch.clamp(Y, min=1)
    d = tempY - F.cosine_similarity(x1, x2) * tempY
   
    m = tempY
    
    v = torch.clamp(m - d, min=0)
    loss = (1 - torch.clamp(Y,max=1)) * 0.5 * d * d + torch.clamp(Y,max=1) * 0.5 * v * v
    return loss.mean()      

# This dataset is for training
class CustomDataset(Dataset):
    # With NFA means to perform negative feature augmentation.
    def __init__(self, dataX, dataY, withNFA):
        self.withNFA = withNFA   
        self.dataX = dataX
        self.dataY = dataY
        
    def __len__(self):        
        return len(self.dataY)
    
    def __getitem__(self, idx):
            
        id1 = idx
        id2 = id1
        
        # 50% to generate two different categories
        if random.random() > 0.5:       
            # Perform NFA
            if self.withNFA and random.random() > 0.5:      
                x1 = self.dataX[id1]
                if random.random()>0.5:
                    x2 = x1 + x1 * 0.5
                else:
                    x2 = x1 - x1 * 0.5
                return x1, x2, 1
            else:
                while self.dataY[id1] == self.dataY[id2]:
                    id2 = random.randint(0, len(self.dataY) - 1) 
                return self.dataX[id1], self.dataX[id2], 1      
        else:               
            while id1 == id2:
                id2 = random.randint(0, len(self.dataY) - 1)    
                if self.dataY[id1] != self.dataY[id2]:
                    id2 = id1
        return self.dataX[id1], self.dataX[id2], 0


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

# Define the device is cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the siamese network
model = Net().to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
trainDataset = CustomDataset(trainDataX, trainDataY, True)
train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True)   

# Train for 10 epochs
bar = tqdm(range(10))
for epoch in bar:
    running_loss = 0
    for x1, x2, y in train_loader:
        x1 = x1.to(device)            
        x2 = x2.to(device) 
        y = y.to(device)
        optimizer.zero_grad()
        x1 = model(x1)
        x2 = model(x2)
   
        loss = constructiveLoss(x1, x2, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        bar.set_description('loss:{}'.format(running_loss))       
    

# a function to find the maximum similarity from the anchor image
def getMaxSimilarity(inp, data):
    maxSimilarity = 0
    for f in data:
        d = cosine_similarity([inp], [f])
        maxSimilarity = max(d[0][0], maxSimilarity)    
    return maxSimilarity


# Now start testing
# Set the face descriptor to detect the face from the video.
descriptor = FaceDescriptor(withAlignment=True, alignmentType='proposed')       
model.eval()

threshold = 0.8

# Create the GNG model for short-term memory system
gng = GNG(512)
anchorImage = []
cap = cv2.VideoCapture('material/corner.mp4')    
while(cap.isOpened()):
    ret, frame = cap.read()
    displayStr= "Face Not Detected"
    if ret == True:              
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        face = descriptor.extractFacial(opencv_image)
                    
        similarityResult = 0
        if face is not None:
            feature = descriptor.extractFacialFeatures(face)
            
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
