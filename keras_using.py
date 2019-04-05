# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:23:43 2019

@author: Administrator
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

#Preparing DataSet
(X_train,y_train),(X_test,y_test)=mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)
'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)

'''
#将每个二维的图像矩阵转换成一个一维的向量，然后再将像素值做归一化
X_train = X_train.reshape(X_train.shape[0], -1) # 等价于X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(X_test.shape[0], -1)    # 等价于X_test = X_test.reshape(10000,784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

#把原来0~9这样的标签，变成长度为10的one-hot向量表示。
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


#构建NN，模型训练
model=Sequential()

model.add(Dense(input_dim=28*28,output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='mse',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=100,nb_epoch=20)

score=model.evaluate(X_test,y_test)
print('Total loss on Testing Set:',score[0])
print('Accuracy of Testing Set:',score[1])

result=model.predict(X_test)


