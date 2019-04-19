# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:22:53 2019

@author: Administrator
"""
import numpy as np
#乘法层的简单实现
class Mullayer:
    
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
    
#加法层
class Addlayer:
    
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x+y
        return out
    
    def backward(slef,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
    
#Rectified Linear Unit
class ReLU:
    
    def __init__(self):
        self.mask=None
        
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
    
class sigmoid:
    
    def __init__(self):
        self.out=None
        
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    
    def backward(self,dout):
        dx=dout*(1-self.out)*self.out
        return dx
    
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
        
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        
        return dx
    
#使用交叉熵误差作为softmax函数的损失函数
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)  #避免溢出
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))

class softmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None
        
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx
        