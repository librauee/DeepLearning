# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:31:13 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

myimg=mpimg.imread('img.jpg')
plt.imshow(myimg)
plt.axis('off')  #不显示坐标轴
plt.show()
print(myimg.shape)
full=np.reshape(myimg,[1,66,56,3])
inputfull=tf.Variable(tf.constant(1.0,shape=[1,66,56,3]))
filter1=tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))
op=tf.nn.conv2d(inputfull,filter1,strides=[1,1,1,1],padding='SAME')
o=tf.cast( ((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)) )*255,tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t,f=sess.run([o,filter1],feed_dict={inputfull:full})
    t=np.reshape(t,[66,56])
    
    plt.imshow(t,cmap='Greys_r')  #白底黑字
    plt.axis('off')
    plt.show()