# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:25:07 2017

@author: ellio
"""
import os
import tensorflow as tf
import numpy as np
import time as t
import datetime

#%%Notes
""" .eval() converts a tensor within a session into its real output.
    so tf.one_hot(X).eval() turns the one_hot tensor into a numpy array as needed,
    before this we need to do 'with sess.as_default()' so all the variables are run
    in the same session
    
    """

#%%Load Data
#file Path for functions

user = "Elliot"
#user = "Seb"

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/Machine Learning data/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\OneDrive\\Documents\\University\\Year 4\\ML chinese caligraphy\\Graphs'
SebLOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
elliotLOGDIR = r'C:/Users/ellio/Anaconda3/Lib/site-packages/tensorflow/tmp/SimpleNetwork/'

if user == "Elliot":
    funcPath = funcPathElliot
    dataPath = dataPathElliot
    savePath = savePathElliot
    LOGDIR = elliotLOGDIR
else:
    funcPath = funcPathSeb
    dataPath = dataPathSeb
    savePath = savePathSeb
    LOGDIR = SebLOGDIR

whichTest = 1

LOGDIR = LOGDIR + str(datetime.date.today()) + '/test-{}'.format(whichTest)
#make a directory
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir("..")

#%%Get the data
#set ration of data to be training and testing
def oneHot(numberList,n):
    oneHotArray=np.zeros((len(numberList),n));
    for j in range(len(numberList)):
        oneHotArray[j][numberList[j]] = 1;
    return oneHotArray;

trainRatio = 0.95


print("splitting data...")
startTime=t.time()
#file to open

fileName="1001-1100C"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")
dataLength=len(labels)
#split the data into training and testing
#train data
trainImages = images[0:int(dataLength*trainRatio)]
trainLabels = labels[0:int(dataLength*trainRatio)]
testImages = images[int(dataLength*trainRatio):dataLength]
testLabels = labels[int(dataLength*trainRatio):dataLength]
labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")



#%%
print("Building network...")
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

numOutputs = 3755
inputDim = 40

def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  with sess.as_default():
      # Setup placeholders, and reshape the data
      x = tf.placeholder(tf.float32, shape=[None, inputDim^2], name="x")
      x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
      tf.summary.image('input', x_image, 3)
      y = tf.placeholder(tf.float32, shape=[None, 3755], name="labels")
    
      if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
      else:
        conv1 = conv_layer(x_image, 1, 64, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
      flattened = tf.reshape(conv_out, [-1, 10 * 10 * 64])
    
    
      if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")
      else:
        embedding_input = flattened
        embedding_size = 10*10*64
        logits = fc_layer(flattened, 10*10*64, 3755, "fc")
    
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
      
      """Initialise variables, key step, can only make tensorflow objects after this"""
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(os.path.join(LOGDIR, hparam))
      writer.add_graph(sess.graph)
      #initialise the tensors for the one hot vectors
      tfTestLabels =  tf.one_hot(testLabels,numOutputs)
    
    #  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #  embedding_config = config.embeddings.add()
    #  embedding_config.tensor_name = embedding.name
    #  embedding_config.sprite.image_path = os.path.join(LOGDIR,'sprite_1024.png')
    #  embedding_config.metadata_path = os.path.join(LOGDIR,'labels_1024.tsv')
    #  # Specify the width and height of a single thumbnail.
    #  embedding_config.sprite.single_image_dim.extend([28, 28])
    #  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
      batchSize = 200
      iterations = 32000
      displayNum = 200
      testNum = 600
      i=0
      print("took ",t.time()-startTime," seconds\n")
      while i<iterations:
          print("ITERATION: ",i,"\n------------------------")
          batchImages = trainImages[i%dataLength:i%dataLength+batchSize]
          #batchLabels = tf.one_hot(trainLabels[i%dataLength:i%dataLength+batchSize],numOutputs)
          batchLabels = tf.one_hot(trainLabels[i%dataLength:i%dataLength+batchSize],numOutputs)
          if i % (displayNum) == 0:
              print("evaluating training accuracy...")
              [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batchImages, y: batchLabels.eval()})
              writer.add_summary(s, i)
              #train_accuracy = accuracy.eval(feed_dict={x: batchImages, y_: batchLabels, keep_prob: 1.0})
          if i%(testNum) == 0 and i!=0:
              print("evaluating test accuracy...")
              sess.run(assignment, feed_dict={x: testImages[:1024], y: tfTestLabels[:1024].eval()})
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
              #test_accuracy = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
              #testAccuracy[int(i/(testNum))]=test_accuracy
          sess.run(train_step, feed_dict={x: batchImages, y: batchLabels.eval()})
          #train_step.run(feed_dict={x: batchImages, y: batchLabels, keep_prob: 0.5})
          i+=batchSize

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