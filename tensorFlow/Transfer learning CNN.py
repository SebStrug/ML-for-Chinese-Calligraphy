# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:12:45 2018

@author: ellio
"""

#%% Imports, set directories, seb
name = 'Admin'
funcPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\'+name+'\\Desktop\\MLChinese\\Saved script files'
workingPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:/Users/'+name+'/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
#%% Imports and paths
import os
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from classDataManip import oneHot,makeDir,Data
#set paths
dataPath, LOGDIR = fF.whichUser("Elliot")
#import modules
import tensorflow as tf
import numpy as np
import time as t
#%%Import the data
print("Importing the data...")
lenInput = 1024
start = t.time()
#training data
labels,bottlenecks=fF.readNPZ(dataPath,\
                                 "1001to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
nextLabels,nextBottlenecks = fF.readNPZ(dataPath,\
                                 "1101to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
labels = np.concatenate((labels,nextLabels),axis=0)
bottlenecks = np.concatenate((bottlenecks,nextBottlenecks),axis=0)
nextLabels,nextBottlenecks = fF.readNPZ(dataPath,\
                                 "1101to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
labels = np.concatenate((labels,nextLabels),axis=0)
bottlenecks = np.concatenate((bottlenecks,nextBottlenecks),axis=0)
trainData = Data(bottlenecks,labels)

#testing data
labels,bottlenecks=fF.readNPZ(dataPath,\
                                 "1001to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
nextLabels,nextBottlenecks = fF.readNPZ(dataPath,\
                                 "1001to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
labels = np.concatenate((labels,nextLabels),axis=0)
bottlenecks= np.concatenate((bottlenecks,nextBottlenecks),axis=0)
nextLabels,nextBottlenecks = fF.readNPZ(dataPath,\
                                 "1001to1100_{}to{}chars".format(oldNumOutputs,newNumOutptus),\
                                 "labels","bottlenecks")
bottlenecks = np.concatenate((labels,nextLabels),axis=0)
bottlenecks = np.concatenate((bottlenecks,nextBottlenecks),axis=0)
testLabels = labels
testBottlenecks=bottlenecks
#delete unused variables
del nextLabels; del nextBottlenecks;
del bottlenecks; del labels
print("Took ",t.time()-start," seconds.")


#%%Build the network
print("Building the net...")
#Reset the graph (since we are not creating this in a function!)


def neural_net(LOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,\
               iterations,trainData,testImages,testLabels,trainRatio):
    tf.reset_default_graph()
    LOGDIR = makeDir(LOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,trainRatio)
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
    x = tf.placeholder(tf.float32, [None, lenInput], name="images")
   
    # Show 4 examples of output images
    y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
    
    
    with tf.name_scope('fc2'):
        """Fully connected layer 2, maps 1024 features to the number of outputs"""
        #1024 inputs, 10 outputs
        W_fc2 = weight_variable([lenInput, numOutputs])
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
#    
    
    #%% Start training!
    print("Starting training!")
    trainData.imagePos=0
    trainData.labelPos=0
    epochLength = int(len(trainData.labels)/trainBatchSize) #no. of iterations per epoch
    whichEpoch = 0
    print("Number of iterations per epoch: {}".format(epochLength))
    displayNum = 30
    testNum = 200
    for i in range(iterations):
        """Check a random value in the batch matches its label"""
        batchImages, batchLabels = trainData.nextImageBatch(trainBatchSize), trainData.nextOneHotLabelBatch(trainBatchSize,numOutputs)

    #    randomIndex = random.randint(0,trainBatchSize-1)
    #    print(display(batchImages.eval()[randomIndex], inputDim),batchLabels.eval()[randomIndex])
        
        if i % displayNum == 0:
            train_accuracy, train_summary =sess.run([accuracy, mergedSummaryOp], \
                         feed_dict={x: batchImages, y_: batchLabels})
            train_writer.add_summary(train_summary, i*trainBatchSize)
            
        if i % testNum == 0:
            print("Testing the net...")
            test_accuracy, test_summary = sess.run([accuracy,mergedSummaryOp], \
                           feed_dict={x: testImages,y_: oneHot(testLabels,numOutputs)})
            test_writer.add_summary(test_summary, i*trainBatchSize)
            saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{}.ckpt".format(learningRate,i,test_accuracy)))
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
        for learning_rate in [1E-4]:
            for trainBatchSize in [128]:      
                iterations = 600*int(len(trainData.labels)/trainBatchSize)
                #LOGDIR, whichTest, numOutputs, learningRate, trainBatchSize, iterations
                neural_net(LOGDIR,whichTest,numOutputs,learning_rate,trainBatchSize,iterations,\
                           trainData,testImages,testLabels,trainRatio)
