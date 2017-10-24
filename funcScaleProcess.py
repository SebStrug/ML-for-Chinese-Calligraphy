# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:44:47 2017

@author: Sebastian
"""
from classImageFunctions import imageFunc as iF
import scipy.misc
import numpy as np


#scaling process
def scaleImage(image,downscaleSize):
    #downscaleSize is the dimensions of the smaller image we want
    """Takes an image, upscales it to an appropriate size,
    adds white space so it forms a square,
    finally downscales it to a useable size"""
    height = image.shape[0]
    width = image.shape[1]
    #we must scale the larger dimension
    upscaleSize = int(downscaleSize * np.ceil(max(height,width)/downscaleSize))
    #if height is larger, scale that dimension
    if height>width:
        upscaledImage = scipy.misc.imresize(image,(upscaleSize,int(width*upscaleSize/height)))
    #if width is larger, scale that dimension
    else:
        upscaledImage = scipy.misc.imresize(image,(int(height*upscaleSize/width),upscaleSize))
    #then, add white space so the image is a square
    newDimension = max(upscaledImage.shape[0],upscaledImage.shape[1])
    whiteImage = iF.extendArray(upscaledImage,newDimension,newDimension)
    #reduce the image to an appropriate size
    blockSize = int(newDimension/downscaleSize)
    reducedImage = iF.downscaleArray(whiteImage,blockSize,blockSize)
    return reducedImage

