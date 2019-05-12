# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:34:11 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

n_input=784
n_hidden_1=256
n_hidden_2=2

x=tf.placeholder(tf.float32,[None,n_input])
zinput=tf.placeholder(tf.float32,[None,n_hidden_2])     #中间节点解码器的输入

weights={
        'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev=0.001)),
        'b1':tf.Variable(tf.zeros([n_hidden_1])),
        'mean_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
        'log_sigma_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
        'w2':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_1],stddev=0.001)),
        'b2':tf.Variable(tf.zeros([n_hidden_1])),
        'w3':tf.Variable(tf.truncated_normal([n_hidden_1,n_input],stddev=0.001)),
        'b3':tf.Variable(tf.zeros([n_input])),
        'mean_b1':tf.Variable(tf.zeros([n_hidden_2])),
        'log_sigma_b1':tf.Variable(tf.zeros([n_hidden_2]))
        }
#784-256-2(mean,log_sigma)-256-784
h1=tf.nn.relu(tf.add(tf.matmul(x,weights['w1']),weights['b1']))
z_mean=tf.add(tf.matmul(h1,weights['mean_w1']),weights['mean_b1'])
z_log_sigma_sq=tf.add(tf.matmul(h1,weights['log_sigma_w1']),weights['log_sigma_b1'])

#高斯分布样本,tf.stack 矩阵拼接
eps=tf.random_normal(tf.stack([tf.shape(h1)[0],n_hidden_2]),0,1,dtype=tf.float32)
z=tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)),eps))
h2=tf.nn.relu(tf.matmul(z,weights['w2'])+weights['b2'])
reconstruction=tf.matmul(h2,weights['w3'])+weights['b3']
h2out=tf.nn.relu(tf.matmul(zinput,weights['w2'])+weights['b2'])
reconstructionout=tf.matmul(h2out,weights['w3'])+weights['b3']

#KL散度
reconstr_loss=0.5*tf.reduce_mean(tf.pow(tf.subtract(reconstruction,x),2.0))
latent_loss=-0.5*tf.reduce_sum(1+z_log_sigma_sq-tf.square(z_mean)-tf.exp(z_log_sigma_sq),1)
cost=tf.reduce_mean(reconstr_loss+latent_loss)
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


training_epochs=50
batch_size=128
display_step=5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs})
        if epoch%display_step==0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c))
    print("Finished")
    print("Results:",cost.eval({x:mnist.test.images}))