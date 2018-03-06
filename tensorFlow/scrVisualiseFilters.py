# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:11:18 2018
A script to output a visulaisation of a models filters, will also eventually use it to deploy a network
and produce an embedding etc.  
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
numOutputs= 100#number of outputs in original network

#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
relTrainDataPath = "Machine learning data/TFrecord"#path of training data relative to datapath in classFileFunc
relSavePath = "savedVisualisation" #path for saved images relative to dataPath
relModelPath = 'TF_record_CNN/Outputs100_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName="LR0.001_Iter27720_TestAcc0.86.ckpt"#name of ckpt file with saved model
SaveName = "CNN_LR0.001_BS128"#name for saved images
#import modules
import tensorflow as tf
import numpy as np
import time as t

numImages = 1

#%% import data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'train'+str(numOutputs)+'.tfrecords')
test_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'test'+str(numOutputs)+'.tfrecords')
#%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
# Initialise all variables
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName+".meta")
saver.restore(sess,'./')

graph = tf.get_default_graph()
#print(graph.get_operations())
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data":True}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,numImages,1,**train_kwargs)
    
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/dropout/keep_prob:0")
conv1Activations = graph.get_tensor_by_name("conv_1/Relu:0")
conv2Activations = graph.get_tensor_by_name("conv_2/Relu:0")
#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("add:0")

print("took ",t.time()-start," seconds\n")
#%%
images,labels=sess.run([train_image_batch,train_label_batch])
layer1Activations=sess.run(conv1Activations,feed_dict={x: images, keep_prob: 1.0})
layer2Activations=sess.run(conv2Activations,feed_dict={x: images, keep_prob: 1.0})