#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:13:19 2020

@author: hk3
CS6910:  Assignment2, part 3 (dataset 2)
"""

import torch
import numpy as np
from Autoencoder import Stacked_AE
from Autoencoder import MLFFNN
import copy

train = []
data_set = []
class_names = ["Bag","Coat", "Pullover", "Sandal", "Sneaker"]

for c in class_names:
    fname = "ds2/"+c+".csv"
    file = open(fname, 'r')
    category = []
    
    for line in file:
        example = []
        example.append([float(x)/255 for x in line.split(',')])    
        example = np.array(example).flatten()
        category.append(example)
        train.append(example)
    data_set.append(category)
    file.close()

train = np.array(train)
dim = np.size(train[0],axis=0)       #28*28 images           
     
true_class = []
n_classes = 5
for i in range(n_classes):
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(i)
    
true_class = np.array(true_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
learning_rate = 0.001
batch_size = 50
compressor = Stacked_AE(3, [dim, 256, 64, 32])
compressor.stack_training(train, epochs, learning_rate, batch_size)

epochs = 10
learning_rate = 0.001
batch_size = 50
classifier = MLFFNN([dim, 256, 64, 32, 16, 5])

# import weights
for i in range(3):
    classifier.layer[i].parameters = copy.deepcopy( compressor.AANNs[i].encoder.parameters)

classifier.train(train, true_class, epochs, learning_rate, batch_size)
op = classifier.get_class(train, true_class)

confusion_matrix = np.zeros([5,5])
#row is actual, column is predicted

for i in range(len(op)):
    confusion_matrix[true_class[i]][op[i]] += 1
