#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:04:14 2020

@author: hk3
CS6910: Assignment2, part 1(A)
"""

import torch
import glob   
import numpy as np
from Autoencoder import MLFFNN
from sklearn.decomposition import PCA

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

# use PCA for dimension reduction
pca_dim = 32
pca = PCA(n_components=pca_dim)
red_train = pca.fit_transform(train)

red_train = np.array(red_train)

device = torch.device("cpu")
learning_rate = 0.001
epochs = 100
batch_size = 50

layer_sizes = [pca_dim, 30, 15, 5]
model = MLFFNN(layer_sizes)
model.train(red_train, true_class, epochs, learning_rate, batch_size)
op = model.get_class(red_train, true_class)

confusion_matrix = np.zeros([5,5])
#row is actual, column is predicted

for i in range(len(op)):
    confusion_matrix[true_class[i]][op[i]] += 1

