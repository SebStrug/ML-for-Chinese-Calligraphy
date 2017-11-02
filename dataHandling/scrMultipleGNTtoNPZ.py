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
dataPath = 'C:\\Users\\ellio\\Documents\\Machine learning data\\test'
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
savePathElliot = 'C:\\Users\\ellio\\Documents\\Machine learning data\\'

#zippedList = ML.createZippedList(dataForSaving[0],'zippedListTest',savePathElliot)

#fF.saveNPZ(savePathElliot,"{}Files-characters-images".format(numFiles),saveLabels=dataForSaving[0],\
     #    saveImages=dataForSaving[5])
       
#fF.saveNPZ(savePathSeb,"{}Files-hotOnes-images.format(numFiles)",\
#           saveLabels=ML.newHotOnes(dataForSaving[0],'zippedListTest',savePathSeb),\
#           saveImages=dataForSaving[5])
fF.saveNPZ(savePathElliot,"{}Files-charNums-images".format(numFiles),\
           saveLabels=[ML.storeCharNumber(i,'zippedListTest',savePathElliot) for i in dataForSaving[0]],\
           saveImages=dataForSaving[5])


