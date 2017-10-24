# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:02:00 2017

@author: Sebastian
"""
import numpy as np
from skimage.measure import block_reduce

#Function to reduce an array to scale it down
#reduceArray
#something about "skimage using an aggregate approach" is important
def reduceArray(array,height,width):
    #this is fine as long as we are not binarizing the data
    #it takes the means of slices
    newArray = block_reduce(array, block_size=(height,width), func=np.mean) 
    #can also do np.max
    newArray = np.ceil(newArray).astype(int) #round up and return integers for each element
    return newArray
