# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:43:52 2017

@author: Sebastian
Back to basics
"""
#%% Imports, set directories, seb
name = 'Admin'
funcPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\'+name+'\\Desktop\\MLChinese\\Saved script files'
workingPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:/Users/'+name+'/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
#%% Imports, set directories, Elliot
#funcPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\dataHandling'
#savePath = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'
#workingPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\tensorFlow'
#LOGDIR = r'C:\\Users\\ellio\\Anaconda3\\Lib\\site-packages\\tensorflow\\tmp\\'
#%%

import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import time as t
import datetime
from PIL import Image
import random
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir(workingPath)
from classDataManip import subSet,oneHot,makeDir,Data,createSpriteLabels


#%%Display a MNIST image
def display(img, inputDim, threshold=200):
    """Run as print(display(image_array))"""
    render = ''
    for i in range(len(img)):
        if i % inputDim == 0:
            render += '\n'
        if img[i] > threshold:
            render += '@'
        else:
            render += '.'
    return render
                
#%%Import the data
print("Importing the data...")
inputDim = 48

#%%Build the network
print("Building the net...")

def neural_net(LOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,\
               iterations,trainData,testImages,testLabels,trainRatio):
    tf.reset_default_graph()
    LOGDIR = makeDir(LOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,trainRatio)
    
    #TFRecords path
    TFRecord_path = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\train.tfrecords'
    feature = {'train/image': tf.FixedLenFeature([], tf.string),\
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([TFRecord_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    """ or do we want to convert these into tf.uint8?"""
    image = tf.decode_raw(features['train/image'], tf.float32)
    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [inputDim, inputDim, 1])
    # Any preprocessing here ...
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    
    
    
    # function to create weights and biases automatically
    # want slightly positive weights/biases for relu to avoid dead nurons
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        tf.summary.histogram("weights", initial)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        tf.summary.histogram("biases", initial)
        return tf.Variable(initial)

    # Define the placeholders for images and labels
    x = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
    x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
    # Show 4 examples of output images
    tf.summary.image('input', x_image, 4)
    y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
    
    with tf.name_scope('fc2'):
        """Fully connected layer 2, maps 1024 features to the number of outputs"""
        #1024 inputs, 10 outputs
        W_fc2 = weight_variable([inputDim**2, numOutputs])
        b_fc2 = bias_variable([numOutputs])
        # calculate the convolution
        y_conv = tf.matmul(x, W_fc2) + b_fc2
        tf.summary.histogram("activations", y_conv)
    
    
    # Calculate entropy on the raw outputs of y then average across the batch size
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(\
                            tf.nn.softmax_cross_entropy_with_logits(\
                                labels=y_, \
                                logits=y_conv))
        tf.summary.scalar("xent",cross_entropy)
        
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # what fraction of bools was correct? Cast to floating point...
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)
    
    # Merge all summary operators
    mergedSummaryOp = tf.summary.merge_all()
    # Create a saver to save these summary operations AND the embedding
    saver = tf.train.Saver()
    
    #%% Open a tensorflow session
    print("Initialising the net...")
    sess = tf.InteractiveSession()
    # Initialise all variables
    tf.global_variables_initializer().run()
    
    # Create writers
    train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
    test_writer.add_graph(sess.graph)   
    
    #%% Start training!
    print("Starting training!")
    trainData.imagePos=0
    trainData.labelPos=0
    epochLength = int(len(trainData.labels)/trainBatchSize) #no. of iterations per epoch
    whichEpoch = 0
    print("Number of iterations per epoch: {}".format(epochLength))
    displayNum = 30
    for i in range(iterations):
        """Check a random value in the batch matches its label"""
        batchImages, batchLabels = trainData.nextImageBatch(trainBatchSize), trainData.nextOneHotLabelBatch(trainBatchSize,numOutputs)

    #    randomIndex = random.randint(0,trainBatchSize-1)
    #    print(display(batchImages.eval()[randomIndex], inputDim),batchLabels.eval()[randomIndex])
        
        if i % displayNum == 0:
            train_accuracy, train_summary =sess.run([accuracy, mergedSummaryOp], \
                         feed_dict={x: batchImages, y_: batchLabels})
            train_writer.add_summary(train_summary, i*trainBatchSize)
            saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{}.ckpt".format(learningRate,i,train_accuracy)))
            
        if i % epochLength == 0 and i != 0:
            whichEpoch += 1
            print("Did {} epochs".format(whichEpoch))
        # run a training step   
        sess.run(train_step,feed_dict={x: batchImages,y_: batchLabels})
    train_writer.close()
    test_writer.close()


#%% Run model function multiple times
whichTest = 3
#trainRatio = 0.8
for numOutputs in [30]:
    for trainRatio in [0.8]:
        trainData,testLabels,testImages = prepareDataSet(numOutputs,trainRatio,CharImages,CharLabels)
        for learning_rate in [1E-3]:
            for trainBatchSize in [512]:      
                iterations = 600*int(len(trainData.labels)/trainBatchSize)
                #LOGDIR, whichTest, numOutputs, learningRate, trainBatchSize, iterations
                neural_net(LOGDIR,whichTest,numOutputs,learning_rate,trainBatchSize,iterations,\
                           trainData,testImages,testLabels,trainRatio)
