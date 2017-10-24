# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:47:11 2017

@author: ellio
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

#Functions to initialise weights to non-zero values 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#functions to perform convolution and pooling operations
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#create place holders for nodes(inputs and labels)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#create variebles for the wieghts and biases
x_image = tf.reshape(x, [-1, 28, 28, 1])
#1st conv layer
W_conv1 = weight_variable([5, 5, 1, 32])#layer weights
b_conv1 = bias_variable([32])#layer bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#layer output
#1st pooling layer
h_pool1 = max_pool_2x2(h_conv1)
#second conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#second pool
h_pool2 = max_pool_2x2(h_conv2)
#flatten 
h_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
#fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_flatten, W_fc1) + b_fc1)
#dropout layer
keep_prob = tf.placeholder(tf.float32)
h_drop1 = tf.nn.dropout(h_fc1, keep_prob)
#fully connected layer 2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_drop1, W_fc2) + b_fc2

#%%TRAINING

#caluclate the average cross entropy across a batch between the predictions y_ and the labels y.
#This is the value to reduce
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# define the training method to update the wieghts 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
#caluclate whether the prediction for each image is correct
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#caluclate the average of all the predictions to get a factional accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # put through all training data and update the weights for each batch
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
    #print testing accuracy for every 100 iterations
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#print testing accuracy
  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



