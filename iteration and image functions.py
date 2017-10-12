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

path = 'C:/Users/Sebastian/Desktop/MLChinese/sample 46.png'
def scaleImage(path):
    im = Image.open(path)
    scaling = np.random.uniform(0.9,1.1) #scales by a random value between 0.9 and 1.1
    scaledSize = [i*scaling for i in im.size] #creates size array
    im.thumbnail(scaledSize, Image.ANTIALIAS)
    im.save('foo.png')

def rotateImage(path):
    im = Image.open(path)
    rotation = np.random.uniform(-10,10)
    im = im.rotate(rotation)
    im.save('foo.png')

def translateImage(path):
    img = Image.open(path)
    xyTranslate = np.random.uniform(-10,10,2)
    a = 1; b = 0;
    c = xyTranslate[0] #left/right (i.e. 5/-5)
    d = 0; e = 1;
    f = xyTranslate[1] #up/down (i.e. 5/-5)
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    img.save('foo.jpg')
    print(xyTranslate)
    
def reduceImage(path):
    im = Image.open(path)
    imSize = im.size
    factor = imSize[0]/40 
    newSize = (40,int(imSize[1]/factor))
    # I downsize the image with an ANTIALIAS filter (gives the highest quality)
    im = im.resize(newSize,Image.ANTIALIAS)
    im.save('C:/Users/Sebastian/Desktop/MLChinese/foo.png',optimize=True,quality=95)