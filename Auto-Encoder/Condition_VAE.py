# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:27:04 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

n_input=784
n_hidden_1=256
n_hidden_2=2
n_labels=10

x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_labels])
zinput=tf.placeholder(tf.float32,[None,n_hidden_2])     #中间节点解码器的输入

#全连接层权重wlab1,blab1,作为输入标签的特征转换
weights={
        'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev=0.001)),
        'b1':tf.Variable(tf.zeros([n_hidden_1])),
        'wlab1':tf.Variable(tf.truncated_normal([n_labels,n_hidden_1],stddev=0.001)),
        'blab1':tf.Variable(tf.zeros([n_hidden_1])),
        'mean_w1':tf.Variable(tf.truncated_normal([n_hidden_1*2,n_hidden_2],stddev=0.001)),
        'log_sigma_w1':tf.Variable(tf.truncated_normal([n_hidden_1*2,n_hidden_2],stddev=0.001)),
        'w2':tf.Variable(tf.truncated_normal([n_hidden_2+n_labels,n_hidden_1],stddev=0.001)),
        'b2':tf.Variable(tf.zeros([n_hidden_1])),
        'w3':tf.Variable(tf.truncated_normal([n_hidden_1,n_input],stddev=0.001)),
        'b3':tf.Variable(tf.zeros([n_input])),
        'mean_b1':tf.Variable(tf.zeros([n_hidden_2])),
        'log_sigma_b1':tf.Variable(tf.zeros([n_hidden_2]))
        }

h1=tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))

hlab1=tf.nn.relu(tf.add(tf.matmul(y, weights['wlab1']), weights['blab1']))

hall1= tf.concat([h1,hlab1],1)#256*2

z_mean = tf.add(tf.matmul(hall1, weights['mean_w1']), weights['mean_b1'])
z_log_sigma_sq = tf.add(tf.matmul(hall1, weights['log_sigma_w1']), weights['log_sigma_b1'])

 # sample from gaussian distribution
eps = tf.random_normal(tf.stack([tf.shape(h1)[0], n_hidden_2]), 0, 1, dtype = tf.float32)
z =tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

zall=tf.concat([z,y],1)
h2=tf.nn.relu( tf.matmul(zall, weights['w2'])+ weights['b2'])
reconstruction = tf.matmul(h2, weights['w3'])+ weights['b3']

zinputall = tf.concat([zinput,y],1)
h2out=tf.nn.relu( tf.matmul(zinputall, weights['w2'])+ weights['b2'])
reconstructionout = tf.matmul(h2out, weights['w3'])+ weights['b3']

#cost
reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, x), 2.0))
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                   - tf.square(z_mean)
                                   - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)



training_epochs = 50
batch_size = 128
display_step = 3



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#取数据
            # Fit training using batch data
            _,c = sess.run([optimizer,cost], feed_dict={x: batch_xs,y:batch_ys})
            #c = autoencoder.partial_fit(batch_xs)
        #显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    print("Finished!")
    
    # 测试
    print ("Result:", cost.eval({x: mnist.test.images,y:mnist.test.labels}))
    
    # 根据图片模拟生成图片
    show_num = 10
    pred = sess.run(
        reconstruction, feed_dict={x: mnist.test.images[:show_num],y: mnist.test.labels[:show_num]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))
    plt.draw()
    
    # 根据label模拟生产图片可视化结果
    show_num = 10
    z_sample = np.random.randn(10,2)
    
    pred = sess.run(
        reconstructionout, feed_dict={zinput:z_sample,y: mnist.test.labels[:show_num]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))
    plt.draw()    
    
    
#    pred = sess.run(
#        z, feed_dict={x: mnist.test.images})
#    #x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#    plt.figure(figsize=(6, 6))
#    plt.scatter(pred[:, 0], pred[:, 1], c=mnist.test.labels)
#    plt.colorbar()
#    plt.show()

#    # display a 2D manifold of the digits
#    n = 15  # figure with 15x15 digits
#    digit_size = 28
#    figure = np.zeros((digit_size * n, digit_size * n))
#    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#    
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):
#            z_sample = np.array([[xi, yi]])
#            x_decoded = sess.run(reconstructionout,feed_dict={zinput:z_sample})
#            
#            digit = x_decoded[0].reshape(digit_size, digit_size)
#            figure[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit
#    
#    plt.figure(figsize=(10, 10))
#    plt.imshow(figure, cmap='Greys_r')
    plt.show()