# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:08:25 2018

@author: Sebastian
"""
from funcs import *
import os
from sklearn import cluster,preprocessing
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#dataPathSeb = "C:\\Users\\Sebastian\\Desktop\\GitHub\\hacknight_1\\data"
#gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#
#
#dataPath = dataPathSeb
#os.chdir(dataPath)
#(10)

#%%
# read in the dataset
im1 = np.load('S2_London.npy')
print('Array dimensions: {}'.format(im1.shape))
pixel_size = 10
plt.figure(figsize=(8,8))
plt.imshow(image_histogram_equalization(im1[:,:,:3]))
# gridlines
plt.grid(color='blue')
plt.show()



#%%
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


#%%
# calculate NDVI
ndvi = calculate_NDVI(im1)

# plot
plt.figure(figsize=(8,8))
plt.imshow(ndvi,'nipy_spectral')
plt.title('NDVI')
plt.colorbar()
plt.show()

#%% Caclulate leaf area index LAI
LAI_one = []
LAI_two = []
for i in ndvi:
    for j in i: 
        LAI_one.append(-0.0897 + 1.424 * j) # R = 0.79; 
        LAI_two.append(0.128 * np.exp(j/0.311)) # R = 0.77
        
LAI_one_2D = np.reshape(LAI_one,(1000,1000))
LAI_two_2D = np.reshape(LAI_one,(1000,1000))
# plot
plt.figure(figsize=(8,8))
plt.imshow(LAI_one_2D,'nipy_spectral')
plt.title('LAI_one')
plt.colorbar()
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(LAI_one_2D-ndvi,'nipy_spectral')
plt.title('LAI_one diff')
plt.colorbar()
plt.show()


# plot
plt.figure(figsize=(8,8))
plt.imshow(LAI_two_2D,'nipy_spectral')
plt.title('LAI_two')
plt.colorbar()
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(LAI_one_2D-ndvi,'nipy_spectral')
plt.title('LAI_two diff')
plt.colorbar()
plt.show()
#%%
def multi_to_2d(arr):
    """ convert image array to list array"""
    s = arr.shape
    return np.ravel(arr,(0)).reshape((-1,s[2]))

im_2D = multi_to_2d(im1)
im_2D.shape
im_2D = preprocessing.robust_scale(im_2D)

# construct a KMeans clustering object
clus = cluster.KMeans(n_clusters=6)

# perform clustering operation
## returns a list of cluster IDs
clusters = clus.fit_predict(im_2D)

# reshape the cluster vector to a
# 2D array for plotting

clusters = clusters.reshape(im1.shape[:-1])
clusters

plt.imshow(clusters,'gist_rainbow')
plt.title('Clusters')
plt.colorbar()
plt.show()