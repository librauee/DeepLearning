# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:45:29 2019

@author: Administrator
"""


import numpy as np
import tensorflow as tf


c=tf.constant(0.0)
g=tf.Graph()
with g.as_default():
    c1=tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)
    
g2=tf.get_default_graph()
print(g2)

tf.reset_default_graph()
g3=tf.get_default_graph()
print(g3)

print(c1.name)
t=g.get_tensor_by_name(name="Const:0")
print(t)

a=tf.constant([[1.0,2.0]])
b=tf.constant([[1.0],[3.0]])
tensor1=tf.matmul(a,b,name='example')
print(tensor1.name)
print(tensor1)
test=g3.get_tensor_by_name("example:0")
print(test)
print(tensor1.op.name)
testop=g3.get_operation_by_name("example")
print(testop)
tt2=g.get_operations()
print(tt2)
tt3=g.as_graph_element(c1)
print(tt3)


