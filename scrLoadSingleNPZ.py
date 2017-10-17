# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:43:26 2017

@author: ellio
"""

import os

#file Path for functions
funcPath = 'C:/Users/ellio/OneDrive/Documents/University/Year 4/ML chinese caligraphy/code/'
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF

#file path for data
dataPath = 'C:/Users/ellio/Desktop/training data/first download/HWDB1.1trn_gnt_P1/'
#file to open
fileName="1001"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")

