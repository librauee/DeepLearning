# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:57:16 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X=np.linspace(-1,1,100)
#print(train_X.shape)
train_y=2*train_X+np.random.randn(*train_X.shape)*0.3
'''
plt.plot(train_X,train_y,'ro',label='Original data')
plt.legend()
plt.show()
'''
#占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")

W=tf.Variable(tf.random_normal([1]),name='Weight')
b=tf.Variable(tf.zeros([1]),name='bias')
z=tf.multiply(X,W)+b
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
lr=0.01
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)

init=tf.global_variables_initializer()
training_epochs=20
display_step=2

def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]


with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],"loss":[]}
    #向模型输入数据
    for epoch in range(training_epochs):
        for x,y in zip(train_X,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        #显示训练中的详细信息
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_y})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch+1)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_y}),"W=",sess.run(W),"b=",sess.run(b))
#print(plotdata)      
    plt.plot(train_X,train_y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"]=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.ylabel('loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()