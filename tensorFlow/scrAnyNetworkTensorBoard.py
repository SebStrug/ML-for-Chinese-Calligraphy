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
#first part of the next line goes one back in the directory
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
from classFileFunctions import fileFunc as fF 
"""Define the user"""
funcPath,dataPath,savePath,rootDIR = fF.whichUser('Seb')
os.chdir("..")

def makeDir(rootDIR,hparam):
    """Makes a directory automatically to save tensorboard data to"""
    testNum = 0
    LOGDIR = rootDIR + str(datetime.date.today()) + '/test-{}'.format(testNum)
    while os.path.exists(LOGDIR):
        testNum += 1
        LOGDIR = rootDIR + str(datetime.date.today()) + '/test-{}'.format(testNum)
    #make a directory
    os.makedirs(LOGDIR)
    return LOGDIR

#%%Get the data
#set ration of data to be training and testing

trainRatio = 0.90


print("splitting data...")
startTime=t.time()
#file to open

fileName="1001-1100C"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")
dataLength=len(labels)
#split the data into training and testing
#train data
#trainImages = images[0:int(dataLength*trainRatio)]
#trainLabels = labels[0:int(dataLength*trainRatio)]
#testImages = images[int(dataLength*trainRatio):dataLength]
#testLabels = labels[int(dataLength*trainRatio):dataLength]
trainImages = images[0:3000]
trainLabels = labels[0:3000]
testImages = images[0:3000]
testLabels = labels[0:3000]
trainLength = len(trainLabels)
testLength = len(testLabels)
labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")



#%%
print("Building network...")
def conv_layer(input, size_in, size_out,kernelSize=5,stride=1, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([kernelSize, kernelSize, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME")
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

numConvOutputs = 64
numOutputs = 3755
inputDim = 40

def mnist_model(LOGDIR, learning_rate,batchSize, hparam):
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  with sess.as_default():
      # Setup placeholders, and reshape the data
      x = tf.placeholder(tf.float32, shape=[None, pow(inputDim,2)], name="x")
      x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
      tf.summary.image('input', x_image, 3)
      y = tf.placeholder(tf.float32, shape=[None, numOutputs], name="labels")
    
      embedding_input = x
      embedding_size = pow(inputDim,2)
      
      conv_1 = conv_layer(x_image,1,numConvOutputs)
      flatten = tf.reshape(conv_1,[-1,20*20*numConvOutputs])
      logits = fc_layer(flatten, 20*20*numConvOutputs, numOutputs, "fc")
    
      with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)
    
      with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(xent)
    
      with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    
      summ = tf.summary.merge_all()
    
      embedding = tf.Variable(tf.zeros([3800, embedding_size]), name="test_embedding")
      assignment = embedding.assign(embedding_input)
      saver = tf.train.Saver()
      
      """Initialise variables, key step, can only make tensorflow objects after this"""
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(os.path.join(LOGDIR, hparam))
      writer.add_graph(sess.graph)
      #initialise the tensors for the one hot vectors
      #tf.TestLabels =  tf.one_hot(testLabels,numOutputs)
    
    #  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #  embedding_config = config.embeddings.add()
    #  embedding_config.tensor_name = embedding.name
    #  embedding_config.sprite.image_path = os.path.join(LOGDIR,'sprite_1024.png')
    #  embedding_config.metadata_path = os.path.join(LOGDIR,'labels_1024.tsv')
    #  # Specify the width and height of a single thumbnail.
    #  embedding_config.sprite.single_image_dim.extend([28, 28])
    #  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
      batchSize = batchSize
      iterations = 3001
      displayNum = 10
      testNum = 3002
#      i=0
#      print("took ",t.time()-startTime," seconds\n")
#      while i<iterations:
#          print("ITERATION: ",i,"\n------------------------")
#          batchImages = trainImages[i%dataLength:i%dataLength+batchSize]
#          #batchLabels = tf.one_hot(trainLabels[i%dataLength:i%dataLength+batchSize],numOutputs)
#          batchLabels = tf.one_hot(trainLabels[i%dataLength:i%dataLength+batchSize],numOutputs)
#          if i % (displayNum) == 0:
#              print("evaluating training accuracy...")
#              [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batchImages, y: batchLabels.eval()})
#              writer.add_summary(s, i)
#              #train_accuracy = accuracy.eval(feed_dict={x: batchImages, y_: batchLabels, keep_prob: 1.0})
#          if i%(testNum) == 0 and i!=0:
#              print("evaluating test accuracy...")
#              sess.run(assignment, feed_dict={x: testImages[:3800], y: tf.TestLabels[:3800].eval()})
#              saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
##              test_accuracy = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
#          sess.run(train_step, feed_dict={x: batchImages, y: batchLabels.eval()})
#          #train_step.run(feed_dict={x: batchImages, y: batchLabels, keep_prob: 0.5})
#          i+=batchSize
          
          
      print("Creating dataset tensors...")
      tensorCreation = t.time()
      #create dataset for training and validation
      tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
      tr_data = tr_data.repeat()
      #take a batch of 128
      tr_data = tr_data.batch(batchSize)
      val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))

      val_data = val_data.shuffle(buffer_size=10000)
      #repeat the test dataset infinitely, so that we can loop over its test
      val_data = val_data.repeat()
      #loop over the test value
      val_data = val_data.batch(testLength)
      print("took {} seconds\n".format(t.time()-tensorCreation))
      # create TensorFlow Iterator object
      iteratorCreation = t.time()
      print("Creating the iterator...")
      #training iterator
      tr_iterator = tr_data.make_initializable_iterator()
      next_image, next_label = tr_iterator.get_next()
      #validation iterator (not really iterator, takes in all test values)
      val_iterator = val_data.make_initializable_iterator()
      next_val_image, next_val_label = val_iterator.get_next()
      print("took {} seconds\n".format(t.time()-iteratorCreation))
      
      print("Initialising the iterator...")
      iteratorInitialisation = t.time()
      sess.run(tr_iterator.initializer)
      sess.run(val_iterator.initializer)  
      print("took {} seconds\n".format(t.time()-iteratorInitialisation))
      
      print(tf.one_hot(next_label,numOutputs).eval())
      print(len(tf.one_hot(next_label,numOutputs).eval()))
      
      for i in range(iterations): #range 2001
          if i % displayNum == 0:
              print('calculating training accuracy... i={}'.format(i))
              [train_accuracy, s] = sess.run([accuracy, summ], \
                  feed_dict={x: next_image.eval(), \
                             y: tf.one_hot(next_label,numOutputs).eval()})
              writer.add_summary(s, i)
          if i % testNum == 0 and i!=0:
              print('did 500, saving')
              sess.run(assignment, \
                       feed_dict={x: next_val_image.eval()[:3800],  \
                                  y: tf.one_hot(next_val_label,numOutputs).eval()[:3800]})
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
          sess.run(train_step, \
                   feed_dict={x: next_image.eval(), \
                              y: tf.one_hot(next_label,numOutputs ).eval()})

def make_hparam_string(learning_rate,batchSize):
  fc_param = "fc=1"
  conv_param = "conv=1"
  return "lr_%.0E,batch_%s,%s,%s" % (learning_rate,batchSize, fc_param, conv_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-7]:
      for batchSize in [128]:

        # Include "False" as a value to try different model architectures
            # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate,batchSize)
        print('Starting run for %s' % hparam)
        LOGDIR = makeDir(rootDIR,hparam)
    	    # Actually run with the new settings
        mnist_model(learning_rate,batchSize, hparam)


if __name__ == '__main__':
  main()