# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:41:20 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#print(mnist.train.images)
tf.reset_default_graph()
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])  

W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))
pred=tf.nn.softmax(tf.matmul(x,W)+b)  
#交叉熵损失函数
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
lr=0.01
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)
training_epochs=50
batch_size=100
display_step=1


saver=tf.train.Saver()
model_path="log/mnist_model.ckpt"
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #运行优化器
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            #计算平均损失
            avg_cost+=c/total_batch
        #显示训练中的详细信息
        if (epoch+1)%display_step==0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
            
    print("Finished! ")
    #测试model1
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  #按行提取最大值的索引
    #计算准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #tf.cast转换成指定类型
    #print(accuracy)
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    #保存模型
    
    save_path=saver.save(sess,model_path)
    print("Model saved in file : %s"%save_path)
'''  
print("starting 2nd session...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #恢复模型变量
    saver.restore(sess,model_path)
    batch_xs,batch_ys=mnist.train.next_batch(2)
    #测试model
    correct_pre=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))
    print("Acc:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
    output=tf.argmax(pred,1)
    
    outputval,predv=sess.run([output,pred],feed_dict={x:batch_xs})
    print(outputval,predv,batch_ys)
    
    im=batch_xs[0]
    im=im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
'''   
    