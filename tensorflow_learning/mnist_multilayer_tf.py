# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:42:15 2019

@author: Administrator
"""
    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#print(mnist.train.images)
tf.reset_default_graph()

lr=0.01
training_epochs=25
batch_size=100
display_step=1

n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10

x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

#创建model

def multilayer_percptron(x,weights,biases):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    out_layer=tf.matmul(layer_2,weights['out'])+biases['out']
    return out_layer

weights={
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
        }
biases={
       'b1':tf.Variable(tf.random_normal([n_hidden_1])),
       'b2':tf.Variable(tf.random_normal([n_hidden_2])),
       'out':tf.Variable(tf.random_normal([n_classes]))
        }

pred=multilayer_percptron(x,weights,biases)
#定义loss和优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

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
    #print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))   #运行tf，与sess.run()相仿,与下一句效果相同
    print("Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    
    '''
    #保存模型
    save_path=saver.save(sess,model_path)
    print("Model saved in file : %s"%save_path)
    '''