# -*- coding: utf-8 -*-
"""
Created on Sun May  5 09:14:01 2019

@author: Administrator
"""

import tensorflow as tf
#[batch,in_height,in_width,in_channels]
input1=tf.Variable(tf.constant(1.0,shape=[1,5,5,1]))
input2=tf.Variable(tf.constant(1.0,shape=[1,5,5,2]))
input3=tf.Variable(tf.constant(1.0,shape=[1,4,4,1]))

#定义卷积核变量
#[filter_height,filter_width,in_channels,out_channels(卷积核个数)]
filter1=tf.Variable(tf.constant([-1.0,0,0,-1],shape=[2,2,1,1]))
filter2=tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,2]))
filter3=tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,3]))
filter4=tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,2]))
filter5=tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,1]))

#stride=[a,h,w,c]
#b表示在样本上的步长默认为1，也就是每一个样本都会进行运算。
#h表示在高度上的默认移动步长为1，这个可以自己设定，根据网络的结构合理调节。
#w表示在宽度上的默认移动步长为1，这个同上可以自己设定。
#c表示在通道上的默认移动步长为1，这个表示每一个通道都会进行运算。

#padding ‘Valid’边缘不填充
op1=tf.nn.conv2d(input1,filter1,strides=[1,2,2,1],padding='SAME')
op2=tf.nn.conv2d(input1,filter2,strides=[1,2,2,1],padding='SAME')
op3=tf.nn.conv2d(input1, filter3, strides=[1,2,2,1], padding='SAME') #1个通道输入，生成3个feature map
op4=tf.nn.conv2d(input2, filter4, strides=[1,2,2,1], padding='SAME') # 2个通道输入，生成2个feature map
op5=tf.nn.conv2d(input2, filter5, strides=[1,2,2,1], padding='SAME') # 2个通道输入，生成一个feature map
vop1=tf.nn.conv2d(input1, filter1, strides=[1,2,2,1], padding='VALID') # 5*5 对于pading不同而不同
op6=tf.nn.conv2d(input3, filter1, strides=[1,2,2,1], padding='SAME') 
vop6=tf.nn.conv2d(input3, filter1, strides=[1,2,2,1], padding='VALID')  #4*4与pading无关



init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print("op1:\n",sess.run([op1,filter1]))
    print("---------------------")
    
    print("op2:\n",sess.run([op2,filter2])) #1-2多卷积核 按列取
    print("op3:\n",sess.run([op3,filter3])) #1-3
    print("------------------")   
    
    print("op4:\n",sess.run([op4,filter4]))#2-2    通道叠加
    print("op5:\n",sess.run([op5,filter5]))#2-1        
    print("------------------")
  
    print("op1:\n",sess.run([op1,filter1]))#1-1
    print("vop1:\n",sess.run([vop1,filter1]))
    print("op6:\n",sess.run([op6,filter1]))
    print("vop6:\n",sess.run([vop6,filter1])) 