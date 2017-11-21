# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:46:20 2017

@author: Sebastian
"""

# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf
import numpy as np
import os
import time as t
import datetime

LOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/mnist_tutorial/'

#%%Load Data
#file Path for functions

#user = "Elliot"
user = "Seb"

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/Machine Learning data/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\OneDrive\\Documents\\University\\Year 4\\ML chinese caligraphy\\Graphs'

if user == "Elliot":
    funcPath = funcPathElliot
    dataPath = dataPathElliot
    savePath = savePathElliot
else:
    funcPath = funcPathSeb
    dataPath = dataPathSeb
    savePath = savePathSeb


LOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
LOGDIR = LOGDIR + str(datetime.date.today()) + '/test-1'

os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir("..")

#%%Get the data
#set ration of data to be training and testing
trainRatio = 0.8
numOutputs = 3755

print("splitting data...")
startTime=t.time()
#file to open

fileName="1001-1100C"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")
dataLength=len(labels)
#split the data into training and testing
#train data
trainLabels = labels[0:int(dataLength*trainRatio)]
trainImages = images[0:int(dataLength*trainRatio)]
#test data
testLabels =  oneHot(labels[int(dataLength*trainRatio):dataLength],numOutputs)
testImages = images[int(dataLength*trainRatio):dataLength]
labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")

#%%
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
  tf.reset_default_graph()
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  if use_two_conv:
    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")
  else:
    conv1 = conv_layer(x_image, 1, 64, "conv")
    conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])


  if use_two_fc:
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    embedding_input = fc1
    embedding_size = 1024
    logits = fc_layer(fc1, 1024, 10, "fc2")
  else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc_layer(flattened, 7*7*64, 10, "fc")

  with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
  assignment = embedding.assign(embedding_input)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(os.path.join(LOGDIR, hparam))
  writer.add_graph(sess.graph)

  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = os.path.join(LOGDIR,'sprite_1024.png')
  embedding_config.metadata_path = os.path.join(LOGDIR,'labels_1024.tsv')
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([28, 28])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

  for i in range(2001):
    #print('did a batch')
    batch = mnist.train.next_batch(100)
    if i % 5 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
    if i % 500 == 0:
      print('did 500, saving')
      sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
  conv_param = "conv=2" if use_two_conv else "conv=1"
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-4]:

    # Include "False" as a value to try different model architectures
    for use_two_fc in [False]:
      for use_two_conv in [False]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)

	    # Actually run with the new settings
        mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
  main()