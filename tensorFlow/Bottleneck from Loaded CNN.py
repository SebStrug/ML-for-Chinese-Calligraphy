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
#set paths
dataPath, LOGDIR = fF.whichUser("Elliot")
#import modules
import tensorflow as tf
import numpy as np
import time as t
#%% import data
print("Importing the data...")
start = t.time()
inputDim = 40
#MNIST data
#fileName="MNIST_data"
#MNISTLabels,MNISTImages=fF.readNPZ(savePath,fileName,"saveLabels","saveImages")

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
#%%
#define images and labels as a subset of the data
#this function splits the data and prepares it for use in the network, can be used to loop
#over several numOutputs
   
 #%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
# Initialise all variables
#tf.global_variables_initializer().run()
loadLOGDIR = os.path.join(LOGDIR,'2017-12-12/Chinese_conv_6/LR1E-3BatchFull')
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph('model.ckpt10.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
#print(graph.get_operations())
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/Placeholder:0")
accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("fc1/Relu:0")

print("took ",t.time()-start," seconds\n")

#%% Start training!
print("Starting.....")
start = t.time()
bottlenecks=np.zeros((dataLength,1024))

for i in range(dataLength):
   
    bottlenecks[i]=sess.run(getBottleneck,feed_dict={x: CharImages[i:i+1], keep_prob: 1.0})
    
    
print("done")
CharLabels=0
CharImages=0
print("took ",t.time()-start," seconds\n")
print(bottlenecks)