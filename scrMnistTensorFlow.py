# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:26:02 2017

@author: ellio
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#create place holders for nodes(inputs and labels)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#create variebles for the wieghts and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#initialise all variables in the seesion
sess.run(tf.global_variables_initializer())

#calculate the outputs
y = tf.matmul(x,W) + b
#%%TRAINING

#caluclate the average cross entropy across a batch between the predictions y_ and the labels y.
#This is the value to reduce
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# define the training method to update the wieghts 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# put through all training data and update the weights for each batch
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
#%%TESTING MODEL
  
#caluclate whether the prediction for each image is correct
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#caluclate the average of all the predictions to get a factional accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print the accuracy
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





