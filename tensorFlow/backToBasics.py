# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:43:52 2017

@author: Sebastian
Back to basics
"""
#%% Imports, set directories
funcPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
workingPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'

import os
import tensorflow as tf
import numpy as np
import time as t
import datetime
from PIL import Image
import random
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir(workingPath)
from classDataManip import subSet

#make a directory to save tensorboard information in 
whichTest = 1
LOGDIR = LOGDIR + str(datetime.date.today()) + '/MNIST_decayedLearning_{}'.format(whichTest)
#make a directory if one does not exist
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
    
    
#%%Import the data
print("Importing the data...")
#MNIST data
#fileName="MNIST_data"
#MNISTLabels,MNISTImages=fF.readNPZ(savePath,fileName,"saveLabels","saveImages")

#Chinese characters data
CharLabels,CharImages=fF.readNPZ(savePath,"1001to1100","saveLabels","saveImages")
nextLabels,nextImages = fF.readNPZ(savePath,"1101to1200","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
nextLabels,nextImages = fF.readNPZ(savePath,"1201to1300","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
del nextLabels; del nextImages;

#define images and labels as a subset of the data
numOutputs = 10
trainRatio = 0.7
images, labels = subSet(numOutputs,CharImages,CharLabels)
dataLength = len(labels) #how many labels/images do we have?
#del MNISTLabels; del MNISTImages; 
del CharLabels; del CharImages; #free up memory

#define the training and testing images
trainImages = images[0:int(dataLength*trainRatio)]
trainLabels = labels[0:int(dataLength*trainRatio)]
testImages = images[int(dataLength*trainRatio):dataLength]
testLabels = labels[int(dataLength*trainRatio):dataLength]
del images; del labels;        

#%%Build the network
print("Building the net...")
#Reset the graph (since we are not creating this in a function!)
tf.reset_default_graph()
inputDim = 40
trainBatchSize = len(trainLabels) #128

# Define variables
x = tf.placeholder(tf.float32, [None, inputDim**2])
W = tf.Variable(tf.zeros([inputDim**2, numOutputs]))
b = tf.Variable(tf.zeros([numOutputs]))
y = tf.matmul(x, W) + b

# Define the loss and optimizer
y_ = tf.placeholder(tf.float32, [None,numOutputs])
# Calculate entropy on the raw outputs of y then average across the batch size
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#%%Create the dataset tensors for training and validation data
print("Creating dataset tensors...")
tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
#tr_data = tr_data.shuffle(buffer_size=100,seed=2)
tr_data = tr_data.repeat()
tr_data = tr_data.batch(trainBatchSize)
val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))
#val_data = val_data.shuffle(buffer_size=100)
val_data = val_data.batch(len(testLabels))

#Create the training and validation iterators over batches
tr_iterator = tr_data.make_initializable_iterator()
tr_next_image, tr_next_label = tr_iterator.get_next()
val_iterator = val_data.make_initializable_iterator()
val_next_image, val_next_label = tr_iterator.get_next()
 
#%% Open a tensorflow session
print("Initialising the net...")
sess = tf.InteractiveSession()
# Initialise all variables
tf.global_variables_initializer().run()
# Initailise the iterator
sess.run(tr_iterator.initializer)

#%%Display a MNIST image
def display(img, width=inputDim, threshold=200):
    """Run as print(display(image_array))"""
    render = ''
    for i in range(len(img)):
        if i % width == 0:
            render += '\n'
        if img[i] > threshold:
            render += '@'
        else:
            render += '.'
    return render

#%% Start training!
print("Starting training!")
epochLength = int(len(trainLabels)/trainBatchSize) #no. of iterations per epoch
whichEpoch = 0
print("Number of iterations per epoch: {}".format(epochLength))
iterations = 400
for i in range(iterations):
    batchImages, batchLabels = tr_next_image, tr_next_label
    #check a random value in the batch
    randomIndex = random.randint(0,trainBatchSize-1)
    print(display(batchImages.eval()[randomIndex]),batchLabels.eval()[randomIndex])
    #print(display(batchImages.eval()[randomIndex]),batchLabels.eval()[randomIndex])
    sess.run(train_step, \
             feed_dict={x: batchImages.eval(), \
                        y_: tf.one_hot(batchLabels,numOutputs).eval()})
    if i%epochLength == 0 and i != 0:
        whichEpoch += 1
        print("Did {} epochs".format(whichEpoch))

#%% Test trained model
print("Testing the net...")
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
testBatchImages, testBatchLabels = val_next_image, val_next_label
print(sess.run(accuracy, \
               feed_dict={x: testBatchImages.eval(), \
                          y_: tf.one_hot(testBatchLabels,numOutputs).eval()}))

