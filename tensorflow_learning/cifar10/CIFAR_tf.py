# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:25:21 2019

@author: Administrator
"""

import cifar10_input
import tensorflow as tf
import pylab

#取数据
batch_size=128
data_dir='/tmp/cifar10_data/cifar-10-batches-bin'
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
image_batch,label_batch=sess.run([images_test,labels_test])
print("__\n",image_batch[0])
print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()

#手动读取cifar
'''
import numpy as np
from scipy.misc import imsave

filename='/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin'
bytestream=open(filename,"rb")
buf=bytestream.read(10000*(1+32*32*31))
bytestream.close()  
  
data = np.frombuffer(buf, dtype=np.uint8)  
data = data.reshape(10000, 1 + 32*32*3)  
labels_images = np.hsplit(data, [1])  
labels = labels_images[0].reshape(10000)  
images = labels_images[1].reshape(10000, 32, 32, 3)  
  
img = np.reshape(images[0], (3, 32, 32)) #导出第一幅图  
img = img.transpose(1, 2, 0)  
  
import pylab 
print(labels[0]) 
pylab.imshow(img)
pylab.show()
'''

#队列中加入协调器
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    
    image_batch, label_batch = sess.run([images_test, labels_test])
    print("__\n",image_batch[0])
    
    print("__\n",label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()
    coord.request_stop()