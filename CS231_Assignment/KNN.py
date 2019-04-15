# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 07:55:42 2019

@author: Administrator
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams['figure.figsize'] = (11.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_CIFAR_batch(filename):
#加载单个文件
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y
def load_CIFAR10_classes(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        labels_name = []
        for i in dict[b'label_names']:
            labels_name.append(i.decode('utf-8'))
    return labels_name
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    labels_name = load_CIFAR10_classes(os.path.join(ROOT, 'batches.meta'))
    return Xtr, Ytr, Xte, Yte, labels_name
data_path = 'E:\git\Git\DeepLearning\CS231_Assignment\cifar-10-batches-py'
X_train, y_train, X_test, y_test,labels_name = load_CIFAR10(data_path)

print('训练集：', X_train.shape)
print('训练集标签：',y_train.shape)
print('测试集：', X_test.shape)
print('测试集标签：', y_test.shape)
print('标签集合：', labels_name)