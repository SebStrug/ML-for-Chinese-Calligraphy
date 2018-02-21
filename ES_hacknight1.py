# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:08:25 2018

@author: Sebastian
"""
from funcs import *
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dataPathSeb = "C:\\Users\\Sebastian\\Desktop\\GitHub\\hacknight_1\\data"
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path



dataPath = dataPathSeb
os.chdir(dataPath)

# read in the dataset
im1 = np.load('S2_London.npy')
print('Array dimensions: {}'.format(im1.shape))
pixel_size = 10
plt.figure(figsize=(8,8))
plt.imshow(image_histogram_equalization(im1[:,:,:3]))
# gridlines
plt.grid(color='blue')
plt.show()

f = plt.figure(figsize=(8,8))

f.add_subplot(221)
plt.imshow(image_histogram_equalization(im1[:,:,0]),'binary_r')
plt.title('Blue')

f.add_subplot(222)
plt.imshow(image_histogram_equalization(im1[:,:,1]),'binary_r')
plt.title('Green')

f.add_subplot(223)
plt.imshow(image_histogram_equalization(im1[:,:,2]),'binary_r')
plt.title('Red')

f.add_subplot(224)
plt.imshow(image_histogram_equalization(im1[:,:,3]),'binary_r')
plt.title('Infrared')
plt.show()



# calculate NDVI
ndvi = calculate_NDVI(im1)

# plot
plt.figure(figsize=(8,8))
plt.imshow(ndvi,'nipy_spectral')
plt.title('NDVI')
plt.colorbar()
plt.show()