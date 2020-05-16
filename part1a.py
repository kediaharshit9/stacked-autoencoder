#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:04:14 2020

@author: hk3
CS6910: Assignment2, part 1(A)
"""

import torch
import glob   
import math
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA

class MLFFNN(nn.Module):
    def __init__(self, dims):
        """
        Parameters
        ----------
        dims : array of size of all layers
        """
        
        super(MLFFNN, self).__init__()
        self.dims = dims
        self.n_layers = len(dims)
        self.layer = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layer.append(nn.Linear(dims[i-1], dims[i]))
            
            
    def forward(self, x):
        for i in range(self.n_layers - 2):
            x = self.layer[i](x)
            x = torch.nn.ReLU(x)
        x = torch.nn.Softmax(self.layer[self.n_layers - 2](x))
        return x
        
    def train(self, train_data, train_results, epochs, learning_rate, batch_size):
        x_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        y_train = Variable(torch.from_numpy(train_results)).type(torch.FloatTensor)
        
        trn = TensorDataset(x_train, y_train)
        trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=True, num_workers=2)
        modulo_factor = math.ceil(np.size(train_data, axis=0)/batch_size)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        print(self)
        
        losses = []
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trn_dataloader):
                data = torch.autograd.Variable(data)
                optimizer.zero_grad()
                pred = self.forward(data)
                loss = loss_func(pred, data)
                losses.append(loss.cpu().data.item())
                loss.backward()
                optimizer.step()
                
                if(batch_idx%modulo_factor==0):
                    print('Train epoch: {}, loss: {}'.format(epoch, loss.cpu().data.item()))
        return
    def get_class(self, dataset):
        x_inp = Variable(torch.from_numpy(dataset)).type(torch.FloatTensor)
        trn = TensorDataset(x_inp)
        dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False)
        preds = []
        for batch_idx, (data) in enumerate(dataloader):
            pred = self.forward(data).detach().numpy().flatten()
            preds.append(pred)
        return preds
    
    
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

# use PCA for dimension reduction
pca_dim = 32
pca = PCA(n_components=pca_dim)
red_train = pca.fit_transform(train)

red_train = np.array(red_train)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
epochs = 100
batch_size = 50

layer_sizes = [pca_dim, 30, 15, 5]
model = MLFFNN(layer_sizes)
#model.train(red_train, true_class, epochs, learning_rate, batch_size)
op = model.get_class(red_train)
