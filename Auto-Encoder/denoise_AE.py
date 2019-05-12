# -*- coding: utf-8 -*-
"""
Created on Sun May 12 9:04:33 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

train_X=mnist.train.images
train_y=mnist.train.labels
test_X=mnist.test.images
test_y=mnist.test.labels

tf.reset_default_graph()

n_input=784
n_hidden_1=256


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_input])
dropout_keep_prob = tf.placeholder("float")

#学习参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_input]))
}

#model
def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_1out = tf.nn.dropout(layer_1, _keep_prob) 
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2'])) 
    layer_2out = tf.nn.dropout(layer_2, _keep_prob) 
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])

reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)


cost = tf.reduce_mean(tf.pow(reconstruction-y, 2))
optm = tf.train.AdamOptimizer(0.01).minimize(cost) 

epochs     = 20
batch_size = 256
disp_step  = 2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print ("Begin training")
    for epoch in range(epochs):
        num_batch  = int(mnist.train.num_examples/batch_size)
        total_cost = 0.
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + 0.3*np.random.randn(batch_size, 784)  #高斯噪声
            feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)
            
        #显示训练日志
        if epoch % disp_step == 0:
            print ("Epoch %02d/%02d average cost: %.6f" 
                   % (epoch, epochs, total_cost/num_batch))
      
    print ("Finished training")
    #可视化
    show_num=10
    test_noisy=test_X[:show_num]+0.3*np.random.randn(show_num,784)
    encode_decode=sess.run(reconstruction,feed_dict={x:test_noisy,dropout_keep_prob:1.})
    f,a=plt.subplots(3,10,figsize=(10,3))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(test_noisy[i],(28,28)))   #加入噪声后
        a[1][i].imshow(np.reshape(test_X[i],(28,28)))       #原始图像
        a[2][i].imshow(np.reshape(encode_decode[i],(28,28)),cmap=plt.get_cmap('gray'))  #去噪后图像
    plt.show()
