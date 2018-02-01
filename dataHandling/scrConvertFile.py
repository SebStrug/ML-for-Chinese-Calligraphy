# -*- coding: utf-8 -*-
"""
"""
#%%Load Data
#file Path for functions

#user = "Seb"
user = "Seb"

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/forConversion/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'


if user == "Elliot":
    funcPath = funcPathElliot
    dataPath = dataPathElliot
    savePath = savePathElliot
elif user == "Seb":
    funcPath = funcPathSeb
    dataPath = dataPathSeb
    savePath = savePathSeb

#%%
import os
import numpy as np
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Extract data
#file path for data
#dataPath = 'C:\\Users\\ellio\\Documents\\training data\\forConversion'
dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\1.0 test'

#%%
#create an array that contains all the bytes
byteInfo,totalFiles = fF.iterateOverFiles(dataPath)
#get file information
dataInfo = fF.infoGNT(byteInfo,totalFiles) #1 for 1 file
dataForSaving = fF.arraysFromGNT(byteInfo,dataInfo)
del byteInfo; del dataInfo

#%%
def checkImage1D(array):
    dimension = int(array.shape[0]**0.5)
    return Image.fromarray(np.resize(array,(dimension,dimension)), 'L')

def checkImage2D(array):
    return Image.fromarray(np.resize(array,(array.shape[0],array.shape[1])), 'L')

#%% New image manipulation formula
def extendArray(array):
    """Generates a new array of zeros with a size defined by the max height and max width,
    with the original array in question in the centre of that array."""
    height = array.shape[0]
    width = array.shape[1]
    largeDim = max(height,width)
    newArray = np.full((largeDim,largeDim),255).astype(np.uint8)
    lowerBound = largeDim//2 - height//2
    leftBound = largeDim//2 - width//2
    newArray[lowerBound:lowerBound+height, leftBound:leftBound+width] = array
    paddedArray = np.pad(newArray,(2,2),'constant',constant_values=(255,255))
    return paddedArray.astype(np.uint8)












