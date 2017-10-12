# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:14:01 2017

@author: Sebastian
"""
import os
import numpy as np


"""
Iterates over all folders in a directory
Finds all the files in all folders
Reads them in, output as fullFile
"""
def iterateOverFiles(path):
    totalFiles = 0
    source = 'C:/Users/Sebastian/Desktop/MLChinese/CASIA/HWDtest2'
    for subdir, dirs, filenames in os.walk(source):
        totalFiles += len(filenames)
    print(totalFiles)
    
    fullFile = [None]*totalFiles
    for subdir, dirs, filenames in os.walk(source):
        for file in filenames:
            fullpath = os.path.join(subdir, file)
            with open(fullpath, 'rb') as openFile:
                fullFile[filenames.index(file)] = openFile.readlines()[0]
                openFile.close()
    return fullFile
