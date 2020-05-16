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
validation = []
data_set = []
class_names = ["Bag","Coat", "Pullover", "Sandal", "Sneaker"]

for c in class_names:
    fname = "ds2/"+c+".csv"
    file = open(fname, 'r')
    category = []
    count = 0
    for line in file:
        example = []
        example.append([float(x)/255 for x in line.split(',')])    
        example = np.array(example).flatten()
        category.append(example)
        count+=1
        if(count<=5000):
            train.append(example)
        if(count >5000):
            validation.append(example)
    data_set.append(category)
    
    file.close()

train = np.array(train)
validation = np.array(validation)

dim = np.size(train[0],axis=0)       #28*28 images           
     
n_classes = 5
true_class = np.zeros(25000, dtype=int)
true_class[5000:9999] = 1
true_class[10000:14999] = 2
true_class[15000:19999] = 3
true_class[20000:24999] = 4
true_class = true_class.flatten()

val_class = np.zeros(5000, dtype=int)
val_class[1000:1999] = 1
val_class[2000:2999] = 2
val_class[3000:3999] = 3
val_class[4000:4999] = 4
val_class = val_class.flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5
learning_rate = 0.001
batch_size = 100
compressor = Stacked_AE(3, [dim, 256, 64, 32])
compressor.stack_training(train, epochs, learning_rate, batch_size)

epochs = 10
learning_rate = 0.001
batch_size = 100
classifier = MLFFNN([dim, 256, 64, 32, 16, 5])

# import weights
for i in range(3):
    classifier.layer[i].parameters = copy.deepcopy(compressor.AANNs[i].encoder.parameters)

classifier.train(train, true_class, epochs, learning_rate, batch_size)
op = classifier.get_class(train, true_class)

confusion_matrix_train = np.zeros([5,5])
#row is actual, column is predicted
for i in range(len(op)):
    confusion_matrix_train[true_class[i]][op[i]] += 1

op_val = classifier.get_class(validation, val_class)
confusion_matrix_val = np.zeros([5,5])
for i in range(len(op_val)):
    confusion_matrix_val[val_class[i]][op_val[i]] += 1
