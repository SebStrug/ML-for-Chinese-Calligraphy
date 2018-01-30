# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:38:17 2018

@author: ellio
"""

#%% Imports, set directories, seb
import os
name = 'Admin'
#funcPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
#savePath = 'C:\\Users\\'+name+'\\Desktop\\MLChinese\\Saved script files'
#workingPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
#LOGDIR = r'C:/Users/'+name+'/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
#%% Imports, set directories, Elliot
dataPath = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'
LOGDIR = r'C:\\Users\\ellio\\Anaconda3\\Lib\\site-packages\\tensorflow\\tmp\\'

gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(gitHubRep)
import tensorflow as tf
import numpy as np
import time as t
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from classDataManip import subSet,oneHot,makeDir,Data,createSpriteLabels

