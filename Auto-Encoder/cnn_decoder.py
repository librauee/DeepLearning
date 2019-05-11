# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:22:21 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

#最大池化
def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net, mask

#4*4----2*2--=2*2 【6，8，12，16】    
#反池化
def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()

    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range

    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

 
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
                        

lr = 0.01
n_conv_1 = 16 # 第一层16个ch
n_conv_2 = 32 # 第二层32个ch
n_input = 784 # MNIST data 输入 (img shape: 28*28)
batchsize = 50

# 占位符
x = tf.placeholder("float", [batchsize, n_input])#输入

x_image = tf.reshape(x, [-1,28,28,1])


# 编码
def encoder(x):
    h_conv1 = tf.nn.relu(conv2d(x, weights['encoder_conv1']) + biases['encoder_conv1'])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights['encoder_conv2']) + biases['encoder_conv2'])  
    return h_conv2,h_conv1

# 解码
def decoder(x,conv1):
    t_conv1 = tf.nn.conv2d_transpose(x-biases['decoder_conv2'], weights['decoder_conv2'], conv1.shape,[1,1,1,1])
    t_x_image = tf.nn.conv2d_transpose(t_conv1-biases['decoder_conv1'], weights['decoder_conv1'], x_image.shape,[1,1,1,1])
    return t_x_image

#tf.truncated_normal
#从截断的正态分布中输出随机值
#生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。

#学习参数
weights = {
    'encoder_conv1': tf.Variable(tf.truncated_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'encoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1)),
    'decoder_conv1': tf.Variable(tf.random_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'decoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1))
}
biases = {
    'encoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'encoder_conv2': tf.Variable(tf.zeros([n_conv_2])),
    'decoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'decoder_conv2': tf.Variable(tf.zeros([n_conv_2])),
}


#输出的节点
encoder_out,conv1 = encoder(x_image)
h_pool2, mask = max_pool_with_argmax(encoder_out, 2)

h_upool = unpool(h_pool2, mask, 2)
pred = decoder(h_upool,conv1)

cost = tf.reduce_mean(tf.pow(x_image - pred, 2))
optimizer = tf.train.RMSPropOptimizer(lr).minimize(cost)
training_epochs = 20 
display_step = 5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    total_batch = int(mnist.train.num_examples/batchsize)
    # 开始训练
    for epoch in range(training_epochs):      
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batchsize)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))

    print("完成!")
    
    #测试
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    print ("Error:", cost.eval({x: batch_xs}))
    #可视化结果
    show_num = 10
    reconstruction = sess.run(
        #pred, feed_dict={x: mnist.test.images[:show_num]})
        pred, feed_dict={x: batch_xs})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        #a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[0][i].imshow(np.reshape(batch_xs[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
    plt.draw()