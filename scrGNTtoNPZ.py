# -*- coding: utf-8 -*-
"""
Spyder Editor
#
This is a temporary script file.
"""
#%%
import os


#file Path for functions
funcPath = 'C:/Users/ellio/OneDrive/Documents/University/Year 4/ML chinese caligraphy/code/'
os.chdir(funcPath)
from saveLoadData import saveNPZ,infoGNT,arraysFromGNT

#file path for data
dataPath = 'C:/Users/ellio/Desktop/training data/first download/HWDB1.1trn_gnt_P1/'
#file to open
fileName="1003-c.gnt"

#get info on gnt file
dataInfo = infoGNT(dataPath,fileName)
data=arraysFromGNT(dataPath,fileName,dataInfo)
saveNPZ(dataPath,"1003",saveLabels=data[0],saveImages=data[1])



