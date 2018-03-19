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
inputDim = 48
inputChars= 3866#number of unique characters in dataset
numOutputs= 100#number of outputs in original network
bottleneckLength = 1024
batchSize = 1024
batchesPerFile = 320
#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
relTrainDataPath = "Machine learning data/TFrecord"#path of training data relative to datapath in classFileFunc
relBottleneckSavePath = "Machine learning data/bottlenecks" #path for saved bottlenecks relative to dataPath
relModelPath = '2_Conv/Outputs100_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName="LR0.001_Iter27720_TestAcc0.86.ckpt"#name of ckpt file with saved model
SaveName = "CNN_LR0.001_BS128"#name for saved bottlenecks
#import modules
import tensorflow as tf
import numpy as np
import time as t

#%% import data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'train'+str(inputChars)+'.tfrecords')
test_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'test'+str(inputChars)+'.tfrecords')
   
 #%% Open a tensorflow session
start1 = t.time()
print("Importing graph.....")
start = t.time()
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Initialise all variables
#tf.global_variables_initializer().run()
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName+".meta")
saver.restore(sess,'./'+modelName)

graph = tf.get_default_graph()
#print(graph.get_operations())
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data": True}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,batchSize,1,**train_kwargs)
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,batchSize,1,**train_kwargs)
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("keep_prob:0")
#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("fc_1/Relu:0")

print("took ",t.time()-start," seconds\n")


#%% extract bottlencks

#train data
print("Extracting Bottlenecks for train data......")
start2=t.time()
trainBottlenecks=np.zeros((0,bottleneckLength))
trainLabels = np.asarray([])

try:
    print("Extracting batches.....")
    start = t.time()
    i = 1
    file=1
    while True:
        trainImageBatch,trainLabelBatch=sess.run([train_image_batch,train_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: trainImageBatch, keep_prob: 1.0})
        trainImageBatch = 0
        trainBottlenecks=np.concatenate((trainBottlenecks,bottleneckBatch),axis=0)
        trainLabels=np.concatenate((trainLabels,trainLabelBatch))
        if i%batchesPerFile == 0 or len(trainLabelBatch) != batchSize:
            print("extraction took ",t.time()-start," seconds\n") 
            print("Saving Train Bottlenecks.....")
            start=t.time()
            fF.saveNPZ(os.path.join(dataPath,relBottleneckSavePath),\
                       "bottleneck_"+SaveName+"_{}to{}chars_train_pt{}".format(numOutputs,inputChars,int(file)),\
                       bottlenecks=trainBottlenecks,labels = trainLabels )
            bottleneckBatch = 0
            trainLabelBatch = 0
            trainBottlenecks=np.zeros((0,bottleneckLength))
            trainLabels = np.asarray([])
            print("Save took ",t.time()-start," seconds\n") 
            print("Extracting batches.....")
            start = t.time()
            file+=1
        i+=1
#        trainBottlenecks=np.concatenate((trainBottlenecks,bottleneckBatch),axis=0)
#        trainLabels=np.concatenate((trainLabels,trainLabelBatch))
except tf.errors.OutOfRangeError:
    print("done")
    print("train data took ",t.time()-start2," seconds\n")

#test data
print("Extracting Bottlenecks for test data......")
start2=t.time()
testBottlenecks=np.zeros((0,bottleneckLength))
testLabels = np.asarray([])
try:
    print("Extracting batch.....")
    start = t.time()
    i = 1
    file=1
    while True:
        testImageBatch,testLabelBatch=sess.run([test_image_batch,test_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: testImageBatch, keep_prob: 1.0})
        testImageBatch = 0
        testBottlenecks=np.concatenate((testBottlenecks,bottleneckBatch),axis=0)
        testLabels=np.concatenate((testLabels,testLabelBatch))
        if i%batchesPerFile == 0 or len(testLabelBatch) != batchSize:
            print("extraction took ",t.time()-start," seconds\n") 
            print("Saving Test Bottlenecks.....")
            start=t.time()
            fF.saveNPZ(os.path.join(dataPath,relBottleneckSavePath),\
                       "bottleneck_"+SaveName+"_{}to{}chars_test_pt{}".format(numOutputs,inputChars,int(file)),\
                       bottlenecks=testBottlenecks,labels = testLabels )
            bottleneckBatch = 0
            testLabelBatch = 0
            testBottlenecks=np.zeros((0,bottleneckLength))
            testLabels = np.asarray([])
            print("Save took ",t.time()-start," seconds\n")   
            print("Extracting batches.....")
            start = t.time()
            file+=1
        i+=1
#        trainBottlenecks=np.concatenate((trainBottlenecks,bottleneckBatch),axis=0)
#        trainLabels=np.concatenate((trainLabels,trainLabelBatch))
except tf.errors.OutOfRangeError:
    print("done")
    print("test data took ",t.time()-start2," seconds\n")

print("Whole process took ",t.time()-start1," seconds")