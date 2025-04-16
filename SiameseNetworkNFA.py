#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:19:19 2025

@author: corner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from tqdm import tqdm
import random

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
    
    
class SiameseModel():
    def __init__(self, device):
        self.device = device
        self.model = Net().to(device)
        
        pass
    
    def train(self,  trainDataX, trainDataY, epoch = 10, withNFA=True):
        # Create dataset and train with NFA
        trainDataset = CustomDataset(trainDataX, trainDataY, withNFA)
        train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True)   
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.model.train()
        bar = tqdm(range(epoch))
        for epoch in bar:
            running_loss = 0
            for x1, x2, y in train_loader:
                x1 = x1.to(self.device)            
                x2 = x2.to(self.device) 
                y = y.to(self.device)
                optimizer.zero_grad()
                x1 = self.model(x1)
                x2 = self.model(x2)
           
                loss = constructiveLoss(x1, x2, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() 
                bar.set_description('loss:{}'.format(running_loss))     
    
    def process(self, feature):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature).to(self.device)
            f = self.model(x[None,...]).cpu().numpy()[0]
        return f
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
