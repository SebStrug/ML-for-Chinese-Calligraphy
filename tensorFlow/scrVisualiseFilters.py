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
from classImageFunctions import imageFunc as iF
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
saveName = "CNN_LR0.001_BS128"#name for saved images
#import modules
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import itertools 

numImages = 128
#%%Func defs
def removeAndSwapAxes(array):
    foo = array
    array=np.swapaxes(np.swapaxes(foo,2,3),1,2)
    
def show_result(features,featureDim, num_out, path, name, show = False, save = False):
    test_images = features

    size_figure_grid = int(num_out/8) #output 25 images in a 5x5 grid
    fig, ax = plt.subplots(8, size_figure_grid, figsize=(8, int(num_out/8)))
    for i, j in itertools.product(range(8), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(num_out):
        i = k // 8
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (featureDim, featureDim)), cmap='gray')
    #label = 'Epoch {0}'.format(num_epoch)
    #fig.text(0.5, 0.04, label, ha='center')
    if save:
        os.chdir(path)
        plt.savefig(name)
    if show:
        plt.show()
    else:
        plt.close() 
#%% import data
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'train'+str(numOutputs)+'.tfrecords')
test_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'test'+str(numOutputs)+'.tfrecords')
#%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
print(0)
# Initialise all variables
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(modelName+".meta")
print(1)
saver.restore(sess,'./'+modelName)

graph = tf.get_default_graph()
print("took ",t.time()-start," seconds\n")
#print(graph.get_operations())

print("Set up data....")
start = t.time()
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data":True}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,numImages,1,**train_kwargs)
print("took ",t.time()-start," seconds\n")

print("Assign operations and placeholders......")  
start = t.time()  
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/dropout/keep_prob:0")
conv1Activations = graph.get_tensor_by_name("conv_1/Relu:0")
conv1Weights = graph.get_tensor_by_name("conv_1/Varaible:0")
conv2Activations = graph.get_tensor_by_name("conv_2/Relu:0")
conv2Weights = graph.get_tensor_by_name("conv_2/Varaible:0")
#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("add:0")
print("took ",t.time()-start," seconds\n")

#%% extract  feature maps
images,labels=sess.run([train_image_batch,train_label_batch])
layer1Activations=sess.run(conv1Activations,feed_dict={x: images, keep_prob: 1.0})
layer1Weights=sess.run(conv1Weights)
layer2Activations=sess.run(conv2Activations,feed_dict={x: images, keep_prob: 1.0})
layer2Weights=sess.run(conv2Weights)
#%%process feature maps
removeAndSwapAxes(layer1Activations)
removeAndSwapAxes(layer2Activations)
fF.saveNPZ(os.path.join(dataPath,relSavePath),"features_raw_{}".format(numImages)+saveName+".npz",\
           images=np.reshape(images,(inputDim,inputDim)),layer1=layer1Activations,weight1=layer1Weights,layer2=layer2Activations,weight2=layer2Weights)
show_result(layer1Activations, inputDim,32,os.path.join(dataPath,relSavePath),"layer1Features.jpg",True,True)
show_result(layer2Activations, inputDim/2,64,os.path.join(dataPath,relSavePath),"layer2Features.jpg",True,True)

