# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:57:32 2018

@author: ellio
"""
#This script is like scrVisualiseFilters but doesnt load the graph, only modifies 
#loaded in activations and weights
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

numImages = 1
#%%Func defs
def removeAndSwapAxes(array):
    foo = array[0]
    array=np.swapaxes(np.swapaxes(foo,1,2),0,1)
    return array
def show_result(features,featureDim, num_out, path, name, show = False, save = False):
    test_images = features

    size_figure_grid = int(num_out/8) #output 25 images in a 5x5 grid
    fig, ax = plt.subplots(8, size_figure_grid, figsize=(8, int(8*8/size_figure_grid)))
    for i, j in itertools.product(range(8), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(num_out):
        i = k // size_figure_grid
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
#%%        
#layer1Activations=removeAndSwapAxes(layer1Activations)
#layer2Activations=removeAndSwapAxes(layer2Activations)
fF.saveNPZ(os.path.join(dataPath,relSavePath),"features_raw_"+saveName+".npz",\
           layer1=layer1Activations,layer2=layer2Activations)
show_result(layer1Activations, inputDim,32,os.path.join(dataPath,relSavePath),"layer1Features.jpg",True,True)
show_result(layer2Activations, int(inputDim/2),64,os.path.join(dataPath,relSavePath),"layer2Features.jpg",True,True)

