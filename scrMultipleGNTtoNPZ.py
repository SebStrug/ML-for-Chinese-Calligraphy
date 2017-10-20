# -*- coding: utf-8 -*-
"""
Spyder Editor
#
This is a temporary script file.
"""
#%%
import os
#import numpy as np


#file Path for functions
#funcPath = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/'
funcPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy'
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF

#file path for data
#elliots path C:\Users\ellio\Desktop\training data\iterate test
<<<<<<< HEAD
dataPath = 'C:/Users/ellio/Desktop/training data/iterate test/'
#file to open
=======
#sebs path C:/Users/Sebastian/Desktop/MLChinese/CASIA/HWDtest2/HWDB1.1tst_gnt
dataPath = 'C:/Users/Sebastian/Desktop/MLChinese/CASIA/HWDtest2/HWDB1.1tst_gnt'
#dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2\\EnglishFiles'
>>>>>>> fbfa86c4b51bad8d6ee4f0ce9d6ea1ab2e2ae672

dataForSaving=0;
data=0;
#get info on gnt file
data,tot = fF.iterateOverFiles(dataPath)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo)
<<<<<<< HEAD
data=0;#delete data in raw byte form 
fF.saveNPZ(dataPath,"1001to1004",saveLabels=dataForSaving[0],saveImages=dataForSaving[1])



=======
#dataInfo = fF.infoGNT(dataPath,fileName)
#data=fF.arraysFromGNT(dataPath,fileName,dataInfo)
#fF.saveNPZ(dataPath,"1001",saveLabels=data[0],saveImages=data[1])
>>>>>>> fbfa86c4b51bad8d6ee4f0ce9d6ea1ab2e2ae672
