# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:02:24 2019

@author: Administrator
"""
import numpy as np

class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None
    def forward(self,x,train_flg=True):
        if train_flg:
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)
        
    def backward(self,dout):
        return dout*self.mask