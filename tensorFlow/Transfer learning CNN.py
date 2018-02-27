# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:12:45 2018

@author: ellio
"""

#%% Imports and paths
import os
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from classDataManip import oneHot,makeDir,Data
#set paths
dataPath, LOGDIR,rawDataPath= fF.whichUser("Elliot")
relBottleneckPath="Machine learning data/bottlenecks"#path of bottlencks relative to dataPath
bottleneckSaveName="CNN_LR0.001_BS128"#bit of save name for bottlenecks specified when they were saved in previous script
#import modules
import tensorflow as tf
import time as t
#set other variables
bottleneckLength = 1024
oldNumOutputs = 10
newNumOutputs = 30      
saveName = "finalLayerCNN"          
#%%Import the data
print("Importing the data...")
start = t.time()
#training data
labels,bottlenecks=fF.readNPZ(os.path.join(dataPath,relBottleneckPath),\
                                 "bottleneck_"+bottleneckSaveName+"_{}to{}chars_train"\
                                 .format(oldNumOutputs,newNumOutputs),\
                                 "labels","bottlenecks")

trainData = Data(bottlenecks,labels.astype(int)-171);

#testing data
labels,bottlenecks=fF.readNPZ(os.path.join(dataPath,relBottleneckPath),\
                                 "bottleneck_"+bottleneckSaveName+"_{}to{}chars_test"\
                                 .format(oldNumOutputs,newNumOutputs),\
                                 "labels","bottlenecks")

testLabels = labels.astype(int)-171
testBottlenecks=bottlenecks
#delete unused variables
del bottlenecks; del labels
print("Took ",t.time()-start," seconds.")


#%%Build the network

#Reset the graph (since we are not creating this in a function!)


def neural_net(LOGDIR,name,whichTest,numOutputs,learningRate,trainBatchSize,\
               epochs,trainData,testImages,testLabels,trainRatio):
    print("Building the net...")
    start=t.time()
    tf.reset_default_graph()
    LOGDIR = makeDir(LOGDIR,name,numOutputs,learningRate,trainBatchSize)
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
    x = tf.placeholder(tf.float32, [None, bottleneckLength], name="images")
   
    # Show 4 examples of output images
    y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
    
    
    with tf.name_scope('fc2'):
        """Fully connected layer 2, maps 1024 features to the number of outputs"""
        #1024 inputs, 10 outputs
        W_fc2 = weight_variable([bottleneckLength, numOutputs])
        b_fc2 = bias_variable([numOutputs])
        # calculate the convolution
        y_conv = tf.matmul(x, W_fc2) + b_fc2
        tf.summary.histogram("activations", y_conv)
    
    
    # Calculate entropy on the raw outputs of y then average across the batch size
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(\
                            tf.nn.softmax_cross_entropy_with_logits(\
                                labels=y_, \
                                logits=y_conv),name="xent")
        tf.summary.scalar("xent",cross_entropy)
        
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # what fraction of bools was correct? Cast to floating point...
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
        tf.summary.scalar("accuracy",accuracy)
    
    # Merge all summary operators
    mergedSummaryOp = tf.summary.merge_all()
    print("Took ",t.time()-start," seconds.")
    
    # Create a saver to save these summary operations AND the embedding
    saver = tf.train.Saver()
    
  
     
    #%% Open a tensorflow session
    print("Initialising the net and creating writers...")
    start=t.time()
    sess = tf.InteractiveSession()
    # Initialise all variables
    tf.global_variables_initializer().run()
    # Create writers
    train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
    test_writer.add_graph(sess.graph)
    print("Took ",t.time()-start," seconds.")
#    
    
    #%% Start training!
    print("Starting training!")
    trainData.imagePos=0
    trainData.labelPos=0
    epochLength = int(len(trainData.labels)/trainBatchSize) #no. of iterations per epoch
    whichEpoch = 0
    print("Number of batches per epoch: {}".format(epochLength))
    displayNum = 30
    testNum = 200
    maxAccuracy = 0.0
    for i in range(epochLength*epochs):
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
            print("testing accuracy:",test_accuracy)
            if test_accuracy > maxAccuracy:
                maxAccuracy = test_accuracy
                saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{}.ckpt".format(learningRate,i,test_accuracy)))
        if i % epochLength == 0 and i != 0:
            whichEpoch += 1
            print("Did {} epochs".format(whichEpoch))
        # run a training step   
        sess.run(train_step,feed_dict={x: batchImages,y_: batchLabels})
    train_writer.close()
    test_writer.close()


#%% Run model function multiple times
whichTest = 1

#trainRatio = 0.8
for numOutputs in [30]:
    for trainRatio in [0.8]:
        for learning_rate in [1E-3]:
            for trainBatchSize in [128]:      
                epochs = 300
                #LOGDIR, whichTest, numOutputs, learningRate, trainBatchSize, iterations
                neural_net(LOGDIR,saveName+"_was{}Out".format(oldNumOutputs),whichTest,numOutputs,learning_rate,trainBatchSize,epochs,\
                           trainData,testBottlenecks,testLabels,trainRatio)
