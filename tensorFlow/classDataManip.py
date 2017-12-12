# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:48:05 2017

@author: Sebastian
"""
import os
import numpy as np
from PIL import Image

#%%Load Data
#first part of the next line goes one back in the directory
#os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
os.chdir('C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling')
#os.chdir('C:\\Users\\ellio\\OneDrive\\Documents\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling')
from classFileFunctions import fileFunc as fF 
"""Define the user"""
funcPath,dataPath,savePath,rootDIR = fF.whichUser('Seb')
#os.chdir("..")

#%%Get the data
#set ration of data to be training and testing

#file to open
dataPath = savePath
fileName="1001to1100"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")

#%%
def createSpriteLabels(images,labels,howMany):
    """Create a nxn image of sprites of the characters"""
    # Don't let the number of sprites be greater than 32 or the projector
    #   won't be able to handle it
    if howMany > 32:
        howMany == 32
    spriteImages = images[0:howMany**2] #array form
    convSpriteImage = [Image.fromarray(np.resize(i,(40,40)), 'L') for i in spriteImages] #convert to image
    dimensions = 40*howMany
    montage = Image.new(mode='RGBA', size=(dimensions, dimensions), color=(0,0,0,0))
    offset_x = offset_y = 0
    row_size = howMany
    i = 0
    for image in convSpriteImage:
        montage.paste(image, (offset_x, offset_y))
        if i % row_size == row_size-1: 
            offset_y += 40
            offset_x = 0
        else:
            offset_x += 40
        i += 1
    montage.save(savePath + '/spriteImages'.format(howMany**2), "png")
    
    spriteLabels = labels[0:32*32]
    with open(savePath + "/spriteLabels.tsv", "w") as record_file:
        #single column meta-data should have no column header!
        for i in spriteLabels:
            record_file.write('{}\n'.format(i))
    return montage, record_file


def checkImage(images,labels):
    """Checks images and labels are correct"""
    allImgs = []
    for i in range(len(labels)):
        allImgs.append([labels[i],Image.fromarray(np.resize(images[i],(40,40)), 'L')])
    return allImgs
    
def subSet(numClasses,images,labels):
    """return subset of characters, i.e. 10 characters with images and labels not 3755"""
    subImages = []
    subLabels = []
    for i in range(len(images)):
        if labels[i] in range(numClasses):
            subImages.append(images[i])
            subLabels.append(labels[i])
    return np.asarray(subImages),np.asarray(subLabels)

def makeDir(rootDIR,fileName,hparam):
    """Makes a directory automatically to save tensorboard data to"""
    testNum = 0
    LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
    while os.path.exists(LOGDIR):
        testNum += 1
        LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
    #make a directory
    os.makedirs(LOGDIR)
    return LOGDIR
        
def normalizePixels(trainImages):
    trainImages = np.asarray(trainImages)
    return trainImages/255

def saveImages(trainImages,trainLabels):
    """Check subset we are using is valid, by matching image to label"""
    os.chdir('C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files\\checkImages')
    for i in range(len(trainImages)):
        tmpImage = Image.fromarray(np.resize(trainImages[i],(40,40)), 'L')
        tmpImage.save('{},label_{}.jpeg'.format(i,trainLabels[i]))