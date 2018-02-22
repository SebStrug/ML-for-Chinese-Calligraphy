# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:31:13 2018

@author: ellio
"""


#%% Imports and paths
import os
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from InputTFRecord import inputs
#set other variables
inputDim = 40
inputChars= 30#number of unique characters in dataset
numOutputs= 10#number of outputs in original network
bottleneckLength = 1024
#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
relTrainDataPath = "Machine learning data/TFrecord"#path of training data relative to datapath in classFileFunc
relBottleneckSavePath = "Machine learning data/bottlenecks" #path for saved bottlenecks relative to dataPath
relModelPath = 'untrainedCNN48x48/Outputs10_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName="LR0.001_Iter180_TestAcc0.176.ckpt"#name of ckpt file with saved model
SaveName = "CNN_LR0.001_BS128"#name for saved bottlenecks
#import modules
import tensorflow as tf
import numpy as np
import time as t

#%% import data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'train'+str(inputChars)+'.tfrecords')
test_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'test'+str(inputChars)+'.tfrecords')
   
 #%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
# Initialise all variables
#tf.global_variables_initializer().run()
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName+".meta")
saver.restore(sess,'./'+modelName)

graph = tf.get_default_graph()
#print(graph.get_operations())

train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,bottleneckLength,1)
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,bottleneckLength,1)
    
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/keep_prob:0")
#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("fc1/bottleneck:0")

print("took ",t.time()-start," seconds\n")


#%% extract bottlencks

#train data
print("Extracting Bottlenecks for train data......")
start = t.time()
trainBottlenecks=np.zeros((0,bottleneckLength))
trainLabels = np.asarray([])
try:
    while True:
        trainImageBatch,trainLabelBatch=sess.run([train_image_batch,train_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: trainImageBatch, keep_prob: 1.0})
        trainBottlenecks=np.concatenate((trainBottlenecks,bottleneckBatch),axis=0)
        trainLabels=np.concatenate((trainLabels,trainLabelBatch))
except tf.errors.OutOfRangeError:
    print("done")
    print("took ",t.time()-start," seconds\n")
print("Saving Train Bottlenecks.....")
start = t.time()
fF.saveNPZ(os.path.join(dataPath,relBottleneckSavePath),"bottleneck_"+SaveName+"_{}to{}chars_train".format(numOutputs,inputChars),\
           bottlenecks=trainBottlenecks,labels = trainLabels )
print("took ",t.time()-start," seconds\n")   

#test data
print("Extracting Bottlenecks for test data......")
start = t.time() 
testBottlenecks=np.zeros((0,bottleneckLength))
testLabels = np.asarray([]) 
try:
    while True: 
        testImageBatch,testLabelBatch=sess.run([test_image_batch,test_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: testImageBatch, keep_prob: 1.0})
        testBottlenecks=np.concatenate((testBottlenecks,bottleneckBatch),axis=0)
        testLabels=np.concatenate((testLabels,testLabelBatch)) 
except tf.errors.OutOfRangeError:
    print("done")
    print("took ",t.time()-start," seconds\n")
print("Saving Test Bottlenecks.....")
start = t.time()
fF.saveNPZ(os.path.join(dataPath,relBottleneckSavePath),"bottleneck_"+SaveName+"_{}to{}chars_test".format(numOutputs,inputChars),\
           bottlenecks=testBottlenecks,labels = testLabels )
print("took ",t.time()-start," seconds\n")