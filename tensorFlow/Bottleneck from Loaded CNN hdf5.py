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
#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
relTrainDataPath = "Machine learning data/TFrecord"#path of training data relative to datapath in classFileFunc
relBottleneckSavePath = "Machine learning data/bottlenecks" #path for saved bottlenecks relative to dataPath
relModelPath = '2_Conv/Outputs100_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName="LR0.001_Iter27720_TestAcc0.86.ckpt"#name of ckpt file with saved model
SaveName = "CNN_LR1e-3_BS128"#name for saved bottlenecks

bottleneckPath = os.path.join(dataPath,relBottleneckSavePath)

#import modules
import tensorflow as tf
import numpy as np
import time as t
import h5py as h



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
#Create hdf5 file

#train data
print("Extracting Bottlenecks for train data......")
start=t.time()
try:
    #open file and create dataset
    f=h.File(bottleneckPath+"/bottleneck_"+SaveName+"_{}to{}chars_train".format(numOutputs,inputChars),'w')
    data=f.create_group('train')
    b=data.create_dataset('bottlenecks',(batchSize,bottleneckLength),chunks=True,maxshape=(None,bottleneckLength))
    l=data.create_dataset('labels',(batchSize,),chunks=True,maxshape=(None,))
     ##load first batch
    trainImageBatch,trainLabelBatch=sess.run([train_image_batch,train_label_batch])
    bottleneckBatch=sess.run(getBottleneck,feed_dict={x: trainImageBatch, keep_prob: 1.0})
    trainImageBatch = 0
    b[-batchSize:] = bottleneckBatch
    bottleneckBatch=0
    l[-batchSize:] = trainLabelBatch
    trainLabelBatch= 0 
    print("Extracting batches.....")
    while True:
        #continue adding batches until all data saved
        trainImageBatch,trainLabelBatch=sess.run([train_image_batch,train_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: trainImageBatch, keep_prob: 1.0})
        trainImageBatch = 0
        b.resize(b.shape[0]+batchSize, axis=0)   
        b[-batchSize:] = bottleneckBatch
        bottleneckBatch=0
        l.resize(l.shape[0]+batchSize,axis=0)
        l[-batchSize:] = trainLabelBatch
        trainLabelBatch= 0 
        
        
except tf.errors.OutOfRangeError:
    print("done")
    print("train data took ",t.time()-start," seconds\n")
    f.close()

#test data
print("Extracting Bottlenecks for test data......")
start=t.time()
try:
    #open file and create dataset
    f=h.File(bottleneckPath+"/bottleneck_"+SaveName+"_{}to{}chars_train".format(numOutputs,inputChars),'w')
    data=f.create_group('test')
    b=data.create_dataset('bottlenecks',(batchSize,bottleneckLength),chunks=True,maxshape=(None, bottleneckLength))
    l=data.create_dataset('labels',(batchSize,),chunks=True,maxshape=(None,))
    ##load first batch
    testImageBatch,testLabelBatch=sess.run([test_image_batch,test_label_batch])
    bottleneckBatch=sess.run(getBottleneck,feed_dict={x: testImageBatch, keep_prob: 1.0})
    testImageBatch = 0
    b[-batchSize:] = bottleneckBatch
    bottleneckBatch=0
    l[-batchSize:] = testLabelBatch
    testLabelBatch = 0 
    print("Extracting batches.....")
    while True:
        #continue adding batches until all data saved
        testImageBatch,testLabelBatch=sess.run([test_image_batch,test_label_batch])
        bottleneckBatch=sess.run(getBottleneck,feed_dict={x: testImageBatch, keep_prob: 1.0})
        testImageBatch = 0
        b.resize(b.shape[0]+batchSize, axis=0)   
        b[-batchSize:] = bottleneckBatch
        bottleneckBatch=0
        l.resize(l.shape[0]+batchSize,axis=0)
        l[-batchSize:] = testLabelBatch
        testLabelBatch = 0 
        
        
except tf.errors.OutOfRangeError:
    print("done")
    print("train data took ",t.time()-start," seconds\n")
    f.close()
print("Whole process took ",t.time()-start1," seconds")