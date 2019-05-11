# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:07:32 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


lr=0.01    

# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
n_hidden_4 = 2
n_input = 784  


x=tf.placeholder("float", [None,n_input])
y=x


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	} 

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.zeros([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.zeros([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.zeros([n_input])),
	}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    #最后一层编码线性输出
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4


#构建模型
encoder_op = encoder(x) 			
y_pred = decoder(encoder_op)	

cost = tf.reduce_mean(tf.pow(y - y_pred, 2))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

training_epochs = 20	
batch_size = 256
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))

    print("Finished training")

    #可视化结果
    show_num = 10
    encode_decode = sess.run(y_pred, feed_dict={x: mnist.test.images[:show_num]})
    #将样本对应的自编码重建图像一并输出比较
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
    #显示数据的二维特征
    aa = [np.argmax(l) for l in mnist.test.labels]                      #将onehot编码转成一般编码
    encoder_result = sess.run(encoder_op, feed_dict={x: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=aa)       
    plt.colorbar()
    plt.show()