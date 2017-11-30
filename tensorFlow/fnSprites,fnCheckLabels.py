# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:48:05 2017

@author: Sebastian
"""
import os
import tensorflow as tf
import numpy as np
import time as t
import datetime
from PIL import Image

#%%Load Data
#first part of the next line goes one back in the directory
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
from classFileFunctions import fileFunc as fF 
"""Define the user"""
funcPath,dataPath,savePath,rootDIR = fF.whichUser('Seb')
os.chdir("..")

#%%Get the data
#set ration of data to be training and testing

#file to open
dataPath = savePath
fileName="CharToNumList_10"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")


#%%
def createSpriteLabels(images,labels):
    """Create a 32x32 image of sprites of the characters"""
    spriteImages = images[0:32*32] #array form
    convSpriteImage = [Image.fromarray(np.resize(i,(40,40)), 'L') for i in spriteImages] #convert to image
    dimensions = 40*32
    montage = Image.new(mode='RGBA', size=(dimensions, dimensions), color=(0,0,0,0))
    offset_x = offset_y = 0
    row_size = 32
    i = 0
    for image in convSpriteImage:
        montage.paste(image, (offset_x, offset_y))
        if i % row_size == row_size-1: 
            offset_y += 40
            offset_x = 0
        else:
            offset_x += 40
        i += 1
    montage.save(savePath + '/sprite_{}'.format(32**2), "png")
    
    spriteLabels = labels[0:32*32]
    with open(savePath + "/spriteLabels.tsv", "w") as record_file:
        #single column meta-data should have no column header!
        for i in spriteLabels:
            record_file.write('{}\n'.format(i))
    return montage, record_file

#%% Check images are correct
def checkImage(images,labels):
    image0 = Image.fromarray(np.resize(images[0],(40,40)), 'L')
    label0 = labels[0]
    image3755 = Image.fromarray(np.resize(images[3755],(40,40)), 'L')
    label3755 = labels[3755]
    print(image0,label0)
    print(image3755,label3755)
    
    allImgs = []
    for i in range(len(labels)):
        if labels[i] == 2604:
            allImgs.append(Image.fromarray(np.resize(images[i],(40,40)), 'L'))
            print(i)
    return allImgs
    