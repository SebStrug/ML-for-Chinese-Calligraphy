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
from readTFRecordOnlyDataset import inputs
#set other variables
inputDim = 40
inputChars= 30#number of unique characters in dataset
numOutputs= 10#number of outputs in original network
bottleneckLength = 1024
#set paths and file names
dataPath, LOGDIR = fF.whichUser("Elliot")
train_tfrecord_filename = os.path.join(dataPath,'train'+str(inputChars)+'.tfrecords')
test_tfrecord_filename = os.path.join(dataPath,'train'+str(inputChars)+'.tfrecords')
modelPath = '2017-12-15\\Chinese_conv_5\\Outputs10_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName='LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'
SaveName = "CNN_LR0.001_BS128"
#import modules
import tensorflow as tf
import numpy as np
import time as t

#%% import data
print("Importing the data...")
start = t.time()

#Chinese characters data
CharLabels,CharImages=fF.readNPZ(dataPath,"1001to1100","saveLabels","saveImages")
nextLabels,nextImages = fF.readNPZ(dataPath,"1101to1200","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
nextLabels,nextImages = fF.readNPZ(dataPath,"1201to1300","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
dataLength = len(CharImages)
del nextLabels; del nextImages;
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

train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,train_batch_size,num_epochs)
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,test_batch_size,0)
    
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/Placeholder:0")
accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("fc1/Relu:0")

print("took ",t.time()-start," seconds\n")


#%% extract bottlencks
print("Starting.....")
start = t.time()
#bottlenecks=np.zeros((dataLength,bottleneckLength))
bottlenecks=np.as_array([])

for i in range(dataLength):
   
    bottleneckBatch=sess.run(getBottleneck,feed_dict={x: CharImages[i:i+1], keep_prob: 1.0})
    bottlenecks=np.concatenate((bottlenecks,bottleneckBatch),axis=0)
    
print("done")
CharLabels=0
CharImages=0
print("took ",t.time()-start," seconds\n")
print(bottlenecks)
print("Saving Bottlenecks.....")
start = t.time()
fF.saveNPZ(dataPath,"bottleneck_"+SaveName+"_{}to{}chars".format(numOutputs,inputChars),\
           bottlenecks=bottlenecks,labels = CharLabels )
print("took ",t.time()-start," seconds\n")