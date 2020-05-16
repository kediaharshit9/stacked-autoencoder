#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:35:19 2020

@author: hk3
CS6910:  Assignment2, part 2
"""

import torch
import glob   
import numpy as np
from Autoencoder import Stacked_AE
from Autoencoder import MLFFNN
import copy

train = []
data_set = []
path = 'ds1/*'   
folders = glob.glob(path)   

for folder in folders:   
    category = []
    class_path = folder + "/*"
    files = glob.glob(class_path)
    for file in files:
        f = open(file, 'r')
        example = []
        for line in f:
            example.append([float(x) for x in line.split(' ')])
            
        example = np.array(example).flatten()
        train.append(example)
        category.append(example)
        f.close()
    data_set.append(np.array(category))

dim = len(train[0])
train = np.array(train)

true_class = []
n_classes = np.size(data_set, axis=0)
class_name = ["coast", "insidecity", "opencountry", "street", "tallbuilding"]
for i in range(n_classes):
    my_class = i
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(my_class)
    
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
classifier = MLFFNN([dim, 256, 64, 32, 5])

# import weights
for i in range(3):
    classifier.layer[i].parameters = copy.deepcopy( compressor.AANNs[i].encoder.parameters)

classifier.train(train, true_class, epochs, learning_rate, batch_size)
op = classifier.get_class(train, true_class)

confusion_matrix = np.zeros([5,5])
#row is actual, column is predicted

for i in range(len(op)):
    confusion_matrix[true_class[i]][op[i]] += 1


     
    
    
    
