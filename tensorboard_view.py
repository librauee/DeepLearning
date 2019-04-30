# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:54:21 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



train_X=np.linspace(-1,1,100)
#print(train_X.shape)
train_y=2*train_X+np.random.randn(*train_X.shape)*0.3


tf.reset_default_graph()
X=tf.placeholder("float")
Y=tf.placeholder("float")

W=tf.Variable(tf.random_normal([1]),name='Weight')
b=tf.Variable(tf.zeros([1]),name='bias')
z=tf.multiply(X,W)+b
tf.summary.histogram('z',z)

cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss function',cost)
lr=0.01
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)

init=tf.global_variables_initializer()
training_epochs=20
display_step=2

plotdata = { "batchsize":[], "loss":[] }

def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

with tf.Session() as sess:
    sess.run(init)
    merged_summary_op=tf.summary.merge_all() #合并所有summar
    summary_writer=tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)
    #向模型中输入数据
    for epoch in range(training_epochs):
        for x,y in zip(train_X,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
    #生成summary
            summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y})
            summary_writer.add_summary(summary_str,epoch)
    #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_y}), "W=", sess.run(W), "b=", sess.run(b))
    #print ("cost:",cost.eval({X: train_X, Y: train_Y}))

    #图形显示
    plt.plot(train_X, train_y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()

    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
    