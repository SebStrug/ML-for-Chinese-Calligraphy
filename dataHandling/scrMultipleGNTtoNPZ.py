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
funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHubPC/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
os.chdir(funcPathElliot)
from classFileFunctions import fileFunc as fF
from classMachineLearning import machineLearning as ML

#file path for data
dataPath = 'C:\\Users\\ellio\\Documents\\training data\\forConversion'
#dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2\\HWDB1.1tst_gnt'
#dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2\\EnglishFiles'
numFiles = len([filenames for subdir, dirs, filenames in os.walk(dataPath)][0])
    

dataForSaving=0;
data=0;
#get info on gnt file
data,tot = fF.iterateOverFiles(dataPath)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo)
data=0;#delete data in raw byte form 



savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
#<<<<<<< HEAD
#zippedList = ML.createZippedList(dataForSaving[0],'zippedListTest',savePathSeb)
#fF.saveNPZ(savePathSeb,"{}Files-characters-images".format(numFiles),saveLabels=dataForSaving[0],\
#
#savePathElliot='C:\\Users\\ellio\\Documents\\training data'
#fF.saveNPZ(savePathElliot,"{}Files-characters-images".format(numFiles),saveLabels=dataForSaving[0],\
#           saveImages=dataForSaving[5])
#=======
savePathElliot = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'

#zippedList = ML.createZippedList(dataForSaving[0],'zippedListTest',savePathElliot)

#fF.saveNPZ(savePathElliot,"{}Files-characters-images".format(numFiles),saveLabels=dataForSaving[0],\
     #    saveImages=dataForSaving[5])
       
#>>>>>>> ebec6f8cc84415c9f4ef1e31e1ee6a9e8062db68
#fF.saveNPZ(savePathSeb,"{}Files-hotOnes-images.format(numFiles)",\
#           saveLabels=ML.newHotOnes(dataForSaving[0],'zippedListTest',savePathSeb),\
#           saveImages=dataForSaving[5])
fF.saveNPZ(savePathElliot,"1001-1100C",\
           saveLabels=[ML.storeCharNumber(i,'charToNumCfiles',savePathElliot) for i in dataForSaving[0]],\
           saveImages=dataForSaving[5])


