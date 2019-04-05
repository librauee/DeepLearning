# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:27:55 2019

@author: Administrator
"""

import tensorflow as tf


hello_dl=tf.constant('Hello,Tensorflow')

a = tf.constant(10)
b= tf.constant(12)
compute_dl=tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(hello_dl))
    print(sess.run(compute_dl))