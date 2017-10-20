# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:39:32 2017

@author: Sebastian
"""
import numpy as np
boo = np.ones((4,3))
maxWidth = 5
maxHeight=5
def extendArray(array, width, height, maxWidth, maxHeight):
    """Generates a new array of zeros with a size defined by the max height and max width,
    with the original array in question in the centre of that array."""
    if width < maxWidth or height < maxHeight:
        newArray = np.zeros((maxHeight,maxWidth))
        lowerBound = maxHeight//2 - height//2
        upperBound = lowerBound+height
        leftBound = maxWidth//2 - width//2
        rightBound = leftBound + width
        newArray[lowerBound:upperBound, leftBound:rightBound] = array
        return newArray
    else:
        return array
print(boo)
print(extendArray(boo,boo.shape[1],boo.shape[0],maxWidth,maxHeight))