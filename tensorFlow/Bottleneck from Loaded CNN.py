# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:31:13 2018

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
from inputTFRecord import inputs
#set other variables
inputDim = 40
inputChars= 30#number of unique characters in dataset
numOutputs= 10#number of outputs in original network
bottleneckLength = 1024
#set paths and file names
dataPath, LOGDIR = fF.whichUser("Elliot")
train_tfrecord_filename = os.path.join(dataPath,'train'+str(inputChars)+'.tfrecords')
test_tfrecord_filename = os.path.join(dataPath,'train'+str(inputChars)+'.tfrecords')
modelPath = 'CNN best'# path of loaded model relative to LOGDIR
modelName='LR0.001_Iter3550_TestAcc0.9211409687995911.ckpt.meta'
SaveName = "CNN_LR0.001_BS128"
#import modules
import tensorflow as tf
import numpy as np
import time as t

#%% import data
print("Importing the data...")
start = t.time()

#Chinese characters data

print("took ",t.time()-start," seconds\n")
   
 #%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
# Initialise all variables
#tf.global_variables_initializer().run()
loadLOGDIR = os.path.join(LOGDIR,modelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName)
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
#print(graph.get_operations())

train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,1024,1)
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,1024,1)
    
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/Placeholder:0")
#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("fc1/Relu:0")

print("took ",t.time()-start," seconds\n")


#%% extract bottlencks

#train data
print("Extracting Bottlenecks for train data......")
start = t.time()
trainBottlenecks=np.as_array([])
trainLabels = np.as_array([])
try:
    while True:
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: train_image_batch.eval(), keep_prob: 1.0})
        trainBottlenecks=np.concatenate((trainBottlenecks,bottleneckBatch),axis=0)
        trainLabels=np.concatentate((trainLabels,train_label_batch.eval()))
except tf.errors.OutOfRangeError:
    print("done")
    print("took ",t.time()-start," seconds\n")
print("Saving Train Bottlenecks.....")
start = t.time()
fF.saveNPZ(dataPath,"bottleneck_"+SaveName+"_{}to{}chars_train".format(numOutputs,inputChars),\
           bottlenecks=trainBottlenecks,labels = trainLabels )
print("took ",t.time()-start," seconds\n")   

#test data
print("Extracting Bottlenecks for train data......")
start = t.time() 
testBottlenecks=np.as_array([])
testLabels = np.as_array([]) 
try:
    while True: 
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: train_image_batch.eval(), keep_prob: 1.0})
        testBottlenecks=np.concatenate((testBottlenecks,bottleneckBatch),axis=0)
        testLabels=np.concatentate((testLabels,test_label_batch.eval()))   
except tf.errors.OutOfRangeError:
    print("done")
    print("took ",t.time()-start," seconds\n")
print("Saving Test Bottlenecks.....")
start = t.time()
fF.saveNPZ(dataPath,"bottleneck_"+SaveName+"_{}to{}chars_test".format(numOutputs,inputChars),\
           bottlenecks=testBottlenecks,labels = testLabels )
print("took ",t.time()-start," seconds\n")