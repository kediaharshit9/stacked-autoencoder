#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:44:20 2020

@author: hk3
CS6910: Assignment2, part 1(B)
"""

import torch
import glob   
import numpy as np
from Autoencoder import AE
from Autoencoder import MLFFNN

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
for i in range(n_classes):
    my_class = np.zeros(n_classes)
    my_class[i] = 1
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(my_class)
    
true_class = np.array(true_class)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use AutoEncoder for dimension reduction
red_dim = 32
model = AE(dim, red_dim)
learning_rate = 0.001
epochs = 10
batch_size = 50
model.train(train, epochs, learning_rate, batch_size)
red_train = model.get_encoding(train) 


layer_sizes = [red_dim, 30, 15, 5]
classifier = MLFFNN(layer_sizes)
classifier.train(red_train, true_class, epochs, learning_rate, batch_size)

op = classifier.get_class(red_train)