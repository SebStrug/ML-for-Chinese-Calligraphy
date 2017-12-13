# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:48:05 2017

@author: Sebastian
"""
import os
import numpy as np
from PIL import Image
import datetime

##%%Load Data
##first part of the next line goes one back in the directory
##os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
#os.chdir('C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling')
##os.chdir('C:\\Users\\ellio\\OneDrive\\Documents\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling')
#from classFileFunctions import fileFunc as fF 
#"""Define the user"""
#funcPath,dataPath,savePath,rootDIR = fF.whichUser('Elliot')
##os.chdir("..")
#
##%%Get the data
##set ration of data to be training and testing
#
##file to open
#dataPath = savePath
#fileName="1001to1100"
#labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")

#%%
def createSpriteLabels(images,labels,howMany,savePath):
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

def oneHot(labelList,numOutputs):
        oneHots = np.zeros((len(labelList),numOutputs))
        oneHots[np.arange(len(labelList)), labelList] = 1
        return oneHots
    
#def makeDir(rootDIR,fileName,hparam):
#    """Makes a directory automatically to save tensorboard data to"""
#    testNum = 0
#    LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
#    while os.path.exists(LOGDIR):
#        testNum += 1
#        LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
#    #make a directory
#    os.makedirs(LOGDIR)
#    return LOGDIR
def makeDir(LOGDIR,whichTest,numOutputs,learningRate,trainBatchSize):
    #make a directory to save tensorboard information in 
    #whichTest = 5
    LOGDIR = LOGDIR + str(datetime.date.today()) + \
                '/Chinese_conv_{}/Outputs{}_LR{}_Batch{}'\
                .format(whichTest,numOutputs,learningRate,trainBatchSize)
    #make a directory if one does not exist
    if not os.path.exists(LOGDIR):
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
        
class Data:
    def __init__(self,images,labels,i=0):
        self.images = images
        self.labels = labels 
        self.i=i
    def nextImageBatch(self,batchSize):
        print("Image Data Position",self.i)
        if batchSize < len(self.images)-self.i:
            oldi=self.i
            self.i+=batchSize
            return self.images[oldi:self.i]
            
        elif batchSize == len(self.images)-self.i:
            oldi=self.i
            self.i=0
            return self.images[oldi:]
        else:
            firstHalf = self.images[self.i:]
            secondHalf = self.images[0:self.i+batchSize-len(self.images)]
            self.i+=batchSize-len(self.images)
            return np.concatenate(firstHalf,secondHalf)
           
            
    
        
    def nextOneHotLabelBatch(self,batchSize,numOutputs):
         print("Label Data Position",self.i)
         if batchSize < len(self.labels)-self.i:
            oldi=self.i
            self.i+=batchSize
            print(self.labels[oldi:self.i])
            print(len(self.labels[oldi:self.i]))
            print(numOutputs)
            return oneHot((self.labels)[oldi:self.i],numOutputs)
            
         elif batchSize == len(self.labels)-self.i:
            oldi=self.i
            self.i=0
            return oneHot(self.labels[oldi:],numOutputs)
         else:
            firstHalf = self.labels[self.i:]
            secondHalf = self.labels[0:self.i+batchSize-len(self.labels)]
            self.i+=batchSize-len(self.labels)
            return oneHot(np.concatenate(firstHalf,secondHalf),numOutputs)
            