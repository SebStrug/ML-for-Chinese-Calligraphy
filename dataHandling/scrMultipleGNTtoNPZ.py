# -*- coding: utf-8 -*-
"""
Spyder Editor
#
This is a temporary script file.
"""
#%%
import os
import numpy as np


#file Path for functions
#funcPath = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/'
funcPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF

#file path for data
#dataPath = 'C:/Users/ellio/Desktop/training data/iterate test/'
dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2\\HWDB1.1tst_gnt'
#dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2\\EnglishFiles'

dataForSaving=0;
data=0;
#get info on gnt file
data,tot = fF.iterateOverFiles(dataPath)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo)

data=0;#delete data in raw byte form 
fF.saveNPZ(dataPath,"1001to10010-c",saveLabels=dataForSaving[0],saveImages=dataForSaving[5])

"""Look at the characters in the data"""
characters = dataForSaving[0]
def hotOnes(characters):
    characters = [ord(i) for i in characters] #convert to numbers
    hotOnes = np.zeros((len(characters), max(characters)+1)) #array of zeros
    #replace a 0 in each vector with a 1 according to the labels
    hotOnes[np.arange(len(characters)), characters] = 1 
    return hotOnes

"""Worried this is quite slow"""
def findVal(zippedList,val):
    return [i[0] for i in zippedList].index(val)
#veeery slow
characters = [ord(i) for i in characters]
zipRange = list(range(len(set(characters))))
zippy = list(zip(list(set(characters)), zipRange))
charVals = [findVal(zippy,i) for i in characters]
def newHotOnes(characters):
    characters = [ord(i) for i in characters] #convert to numbers
    hotOnes = np.zeros((len(characters),len(set(characters)))) #array of zeros (10*3755),3755
    hotOnes[np.arange(len(characters)), charVals] = 1
    return hotOnes

fF.saveNPZ(dataPath,"1001to10010-c",saveLabels=newHotOnes(dataForSaving[5]),saveImages=dataForSaving[5])
