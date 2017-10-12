# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:01:17 2017

@author: ellio
"""
import numpy as np
import os
from PIL import Image
from collections import namedtuple

def byteToInt(byte,byteOrder='little'):
    return int.from_bytes(byte, byteorder=byteOrder)
#
#Function that reads the NPZ file 'name' at 'path' and returns the file object
def readNPZ(path,name,labelName,imageName):
    fullName = f"{path}{name}"
    with open(fullName, 'rb') as ifs:
        fileNPZ = np.load(ifs)
        return fileNPZ[labelName],fileNPZ[imageName]
    
#function that saves a list of arrays 'namedArrays' to the file 'name' at
#location 'path' with the option of compression 'compressed'      
def saveNPZ(path,name,compressed=False,**namedArrays):
    fullName = f"{path}{name}"
    with open(fullName, 'wb') as ofs:
        if compressed:
            np.savez_compressed(ofs,**namedArrays)
        else:
            np.savez(ofs,**namedArrays)
    ofs.close();
    

#%% functions for converting and manipulating images
def arrayToImage(array,height,width):
    np.reshape(array,(height,width),'C')#resize it
    print(array.shape,'\n')
    img = Image.fromarray(array);
    return img;

def resizeImage(image,newWidth,newHeight ):
    newim=Image.new("L",(newWidth,newHeight))
    newim.paste(image,(int((newWidth-image.size[0])/2),int((newHeight-image.size[1])/2)))
    return newim

def PIL2array(image):
    return np.array(image.getdata(), np.uint8)
#%%
def infoGNT(path,name):
    #set path and open file
    os.chdir(path)
    with open(name, 'rb') as f:
        fullFile = f.readlines()[0];
    f.close();
    numSamples = 0;
    totalSize = 0;
    maxWidth=0;
    maxHeight=0;
    position = 0;
    while position < len(fullFile):
        sampleSize = byteToInt(fullFile[position:position+4]);
        maxWidth = max(byteToInt(fullFile[position+6:position+8]),maxWidth)
        maxHeight = max(byteToInt(fullFile[position+8:position+10]),maxHeight)
        numSamples+=1;
        position += sampleSize
        totalSize +=sampleSize;
    infoStruct = namedtuple("myStruct","numSamples maxHeight, maxWidth, totalSize")
    info = infoStruct(numSamples,maxHeight,maxWidth,totalSize)
    print (info)
    return info

def arraysFromGNT(path,name,info):
    os.chdir(path)
    with open(name, 'rb') as f:
        fullFile = f.readlines()[0];
    f.close();
    #create arrays to store data read in
    character = np.zeros(info.numSamples,np.unicode);
    images = np.zeros((info.numSamples,info.maxWidth*info.maxHeight))
    #place data into arrays
    position = 0;
    for i in range(0,info.numSamples-1):
        sampleSize = byteToInt(fullFile[position:position+4]);
        character[i] = fullFile[position+4:position+6].decode('gb2312');
        width = byteToInt(fullFile[position+6:position+8]);
        height = byteToInt(fullFile[position+8:position+10]);
        image = np.zeros((height,width))
        for j in range(0,height):
            for k in range(0,width):
                image[j][k]=fullFile[position+10+j*width+k];
        position +=sampleSize;
        print(i)
        print('character',character[i])
        im = arrayToImage(image,height,width)
        imResize=resizeImage(im,info.maxWidth,info.maxHeight)
        images[i] = PIL2array(imResize);
        
    return [character,images,info.maxHeight,info.maxWidth]
    

  
    
