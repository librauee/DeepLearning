# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:55:42 2019

@author: Administrator
"""

import numpy as np
from scipy import special

class neuralNetwork(object):
    #initialize the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        #self.wih=np.random.rand(self.hnodes,self.inodes)-0.5
        #self.who=np.random.rand(self.onodes,self.hnodes)-0.5
        #means,Variance,shape of array
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #sigmoid function
        self.activation_function=lambda x:special.expit(x)
        pass
    
    #train the neural network
    def train(self,inputs_list,targets_list):
        #convert inputs list and targets list  to 2d array
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        self.who +=self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),
                                 np.transpose(hidden_outputs))
        self.wih +=self.lr*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                                 np.transpose(inputs))
        pass
    
    
    #query the neural network
    def query(self,inputs_list):

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
input_nodes=3
hidden_nodes=3
output_nodes=3

learning_rate=0.3
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
n.train([1,0.5,-0.5],[0.8,0.6,0.6])
print(n.query([1,0.5,-0.5]))

'''
a1=np.random.rand(3,3)  #0-1的随机值
a2=a1-0.5               #-0.5-0.5的随机值
'''
