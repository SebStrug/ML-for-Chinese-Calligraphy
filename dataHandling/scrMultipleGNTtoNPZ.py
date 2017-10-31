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
funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
os.chdir(funcPathSeb)
from classFileFunctions import fileFunc as fF
from classMachineLearning import machineLearning as ML

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

savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
fF.saveNPZ(savePathSeb,"1001to1002-c",saveLabels=dataForSaving[0],\
           saveImages=dataForSaving[5])
fF.saveNPZ(savePathSeb,"hotOnes",\
           saveLabels=ML.newHotOnes(dataForSaving[0],'zippedListTest',savePathSeb),\
           saveImages=dataForSaving[5])
fF.saveNPZ(savePathSeb,"charsAsNums",\
           saveLabels=[ML.storeCharNumber(i,'zippedListTest',savePathSeb) for i in dataForSaving[0]],\
           saveImages=dataForSaving[5])


