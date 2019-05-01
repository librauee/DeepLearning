# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:01:07 2019

@author: Administrator
"""

import tensorflow as tf
global_step=tf.Variable(0,trainable=False)
initial_lr=0.1
lr=tf.train.exponential_decay(initial_lr,global_step=global_step,decay_steps=10,decay_rate=0.9)
opt=tf.train.GradientDescentOptimizer(lr)
add_global=global_step.assign_add(1) #定义一个op,使global_step加1完成计步
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(lr))
    for i in range(20):
        g,rate=sess.run([add_global,lr])
        print(g,rate)