#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:20:41 2025

@author: corner
"""

import numpy as np
import cv2
from FaceDscriptor import FaceDescriptor

class GNG:
    threshold = 0.4
    nodeList = []    
    adjacencyList = []
    
    L1 = 0.2
    L2 = 0.02
    maxAge = 30
    newNodeCount = 30
    
    # Difine the feature dimension of a single node.
    def __init__(self, feature_number):
        self.feature_number = feature_number
        
        self.nodeList.append(np.append(np.random.randn(feature_number),0))
        self.nodeList.append(np.append(np.random.randn(feature_number),0))    
        self.adjacencyList.append([0,1,0])
        
        self.nodeList = np.asarray(self.nodeList)
        self.adjacencyList = np.asarray(self.adjacencyList)
        
        

    # Initaialize the two nodes using sample
    def initializing(self, w1, w2):
        assert(len(w1) != self.feature_number, "Invalid Feature Dimension")
        assert(len(w2) != self.feature_number, "Invalid Feature Dimension")
        
        self.nodeList = []
        self.adjacencyList = []
        self.nodeList.append(np.append(w1.copy(),0))
        self.nodeList.append(np.append(w2.copy(),0))    
        self.adjacencyList.append([0,1,0])
        
        self.nodeList = np.asarray(self.nodeList)
        self.adjacencyList = np.asarray(self.adjacencyList)
        
        
    # learn the input data.
    def pushData(self, w):
        
        if len(self.nodeList)  < 2:
            print("Error, must initialize two nodes")
            return
        
        W = self.nodeList
        E = self.adjacencyList
        
        
        # Perform Consine Similarity Distance
        d = np.sum(W[:,:-1] * w, axis=1)     
        a = np.linalg.norm(W[:,:-1], axis=1)
        b = np.linalg.norm(w, axis=0)
        similarity = d / (a * b)
        dist = 1 - similarity

        # Perform sorting, find the nearest node.
        idx = np.argpartition(dist, 1)
        s1 = idx[0]
        s2 = idx[1]
        
        # If first nearest and second nearest is not connected, connect it.
        t1 = np.logical_and(E[:,0] == s1, E[:,1] == s2)
        t2 = np.logical_and(E[:,1] == s1, E[:,0] == s2)
        t = np.logical_or(t1,t2)
        if True in t:# If previously is connected, then set the age become 0.
            E[t==True, 2] = 0
        else: # If first nearest and second nearest is not connected, connect it.
            E = np.vstack((E,[s1,s2,0]))
       
        # Increase the age of all connected nodes
        if len(E) > 0:
            E[np.logical_or(E[:,0] == s1, E[:,1] == s1),2] += 1
        
        # Increase the error of the nearest node.
        W[s1,-1] += dist[s1]**2
                
        # Perform learning on the nearest node towards the data.
        W[s1,:-1] += self.L1 * (w - W[s1,:-1])
                
        # Perform the neighbor learning on the the neaest node towards the data.
        if len(E) > 0:
            connectedNodes = np.unique(np.concatenate((E[E[:,0] == s1,1], E[E[:,1] == s1,0])))        
            W[connectedNodes,:-1] += self.L2 * (w - W[connectedNodes,:-1])
        
        
        
        
        
        # Remove those edge is too old.
        E = E[E[:,2] < self.maxAge]
        
        # Remove those node that is isolated.
        toRepeat = True
        while toRepeat: # Make sure all isolated node been removed.
            toRepeat = False
            for v in range(len(W)):
                if v not in E[:,:2]:
                    E[E[:,0] > v,0] -= 1
                    E[E[:,1] > v,1] -= 1
                    W = np.delete(W, v, axis=0)
                    toRepeat = True
                    break
            
            
        # Reduce all node error by half.
        W[:,-1] *= 0.5
        
        # create new node after interval
        self.newNodeCount -= 1
        if self.newNodeCount <= 0 and len(W) < 10:
            self.newNodeCount = 30
            q1 = np.argmax(W[:,-1])        
            connectedNodes = np.unique(np.concatenate((E[E[:,0] == q1,1], E[E[:,1] == q1,0])))    
            q2 = connectedNodes[np.argmax(W[connectedNodes, -1])]
            q3 = len(W)
            new_w = (W[q1] + W[q2]) * 0.5      
            W = np.vstack((W,new_w))        
            W[q1, -1] *= 0.5
            W[q2, -1] *= 0.5
            W[q3, -1] = W[q1,-1]        
            E = np.vstack((E,[q1,q3,0]))
            E = np.vstack((E,[q2,q3,0]))
        
        self.nodeList = W
        self.adjacencyList = E
       

    def getMaxSimilarity(self, w):
        W = self.nodeList
       
        d = np.sum(W[:,:-1] * w, axis=1)
        a = np.linalg.norm(W[:,:-1], axis=1)
        b = np.linalg.norm(w, axis=0)
        similarity = d / (a * b)
        return max(similarity)
            
        
    
# Uncomment below for testing purpose.
# #Create GNG Network
# gng = GNG(512)

# # # Create the descriptor
# descriptor = FaceDescriptor( withAlignment=True, alignmentType='proposed')

# # Read the image
# img = cv2.imread('material/test_face.jpg')
# # must convert to RGB arrangement
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# # Extract the facial image from the image
# result = descriptor.extractFacial(img)

# # Extract the facial feature
# feature = descriptor.extractFacialFeatures(result)

# # Initializing the node, you can use two different image.
# f1 = feature + np.random.rand(len(feature))
# f2 = feature - np.random.rand(len(feature))
# gng.initializing(f1, f2)

# # find max similarity
# similarity = gng.getMaxSimilarity(feature)
# print("Similarity Before learning",similarity)

# # learn the input 20 times
# # why learn 20 times, because the initializing input is random
# for i in range(20):
#     gng.pushData(feature)

# # find max similarity
# similarity = gng.getMaxSimilarity(feature)
# print("Similarity after learning",similarity)







        
        
        
        
        