# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:01:50 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

with open('mnist_dataset/mnist_train_100.csv','r',encoding='utf-8') as f: 
    data_list=f.readlines()
    
'''
all_values=data_list[0].split(',')
#asfarray: change string to number,and set up an array
image_array=np.asfarray(all_values[1:]).reshape(28,28)

#plt.imshow(image_array,cmap='Greys',interpolation='None')
scaled_input=np.asfarray(all_values[1:])/255.0*0.99 + 0.01
#print(scaled_input)


#example:
    
onodes=10
targets=np.zeros(onodes)+0.01
targets[int(all_values[0])]=0.99
print(targets)

#[0.99 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]

'''
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
    
    
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5
training_data_list=data_list

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []
with open('mnist_dataset/mnist_train_100.csv','r',encoding='utf-8') as f: 
    test_data_list=f.readlines()

# go through all the records in the test data set
for record in test_data_list:

    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # np.argmax() : the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

#performance =  0.96   when lr=0.2
#performance =  0.89   when lr=0.1