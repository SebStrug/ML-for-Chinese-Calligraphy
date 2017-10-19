# -*- coding: utf-8 -*-
"""
Spyder Editor
#
This is a temporary script file.
"""
#%%
import os


#file Path for functions
funcPath = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/'
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF

#file path for data
#elliots path C:\Users\ellio\Desktop\training data\iterate test
dataPath = 'C:/Users/ellio/Desktop/training data/iterate test/'
#file to open
fileName="1001-c.gnt"

#get info on gnt file
data,tot = fF.iterateOverFiles(dataPath)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo)
#dataInfo = fF.infoGNT(dataPath,fileName)
#data=fF.arraysFromGNT(dataPath,fileName,dataInfo)
#fF.saveNPZ(dataPath,"1001",saveLabels=data[0],saveImages=data[1])



