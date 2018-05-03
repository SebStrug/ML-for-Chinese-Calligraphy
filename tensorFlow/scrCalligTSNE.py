# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:09:56 2018

@author: ellio
"""

#%% Imports and paths
import os
import scipy as sp
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#gitHubRep = 'C:/Users/Sebastian/Desktop/GitHub/ML-for-Chinese-Calligraphy'
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
from classImageFunctions import imageFunc as iF
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from InputTFRecord import inputs
#set other variables
inputDim = 48
numOutputs= 3866#number of outputs in original network
numImages=200

#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
#dataPath = 'C:/Users/Sebastian/Desktop/MLChinese'
relTrainDataPath = "Machine learning data/TFrecord" #path of training data relative to datapath in classFileFunc
#relTrainDataPath = 'CASIA/1.0'
relSavePath = "savedVisualisation" #path for saved images relative to dataPath
#relSavePath = 'Visualising_filters'
relModelPath = 'TF_record_CNN/Outputs100_LR0.001_Batch128'# path of loaded model relative to LOGDIR
#relModelPath = '2conv_100Train_TransferOriginal/Outputs100_LR0.001_Batch128'
modelName = 'LR0.001_Iter20520_TestAcc0.845.ckpt'#name of ckpt file with saved original model

#data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'calligraphy_greyscaled.tfrecords')


#import rest of the modules modules
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import itertools
import PIL.ImageOps
from PIL import Image, ImageDraw, ImageFont
#data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'calligraphy_greyscaled.tfrecords')
#%%load graph
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
print("Initialising all variables...")
# Initialise all variables
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName+".meta")

print("Restoring the session...")
saver.restore(sess,'./'+modelName)

print("Getting default graph...")
graph = tf.get_default_graph()
print("took ",t.time()-start," seconds\n")
#print(graph.get_operations())

print("Set up data....")
start = t.time()
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data":False}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,numImages,1,**train_kwargs)
print("took ",t.time()-start," seconds\n")

print("Assign operations and placeholders......")  
start = t.time()  
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/dropout/keep_prob:0")

getBottleneck = graph.get_tensor_by_name("dropout/dropout/mul:0")
print("took ",t.time()-start," seconds\n")