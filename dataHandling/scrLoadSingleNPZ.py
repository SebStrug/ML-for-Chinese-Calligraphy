# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:43:26 2017

@author: ellio
"""

import os

#file Path for functions
funcPath = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/'
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF

#file path for data
dataPath = 'C:/Users/ellio/Desktop/training data/iterate test/'
#file to open
fileName="1001to1004"
labels,images=fF.readNPZ(fileName,"saveLabels","saveImages")

