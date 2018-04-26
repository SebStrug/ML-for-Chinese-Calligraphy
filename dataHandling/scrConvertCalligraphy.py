# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:25:38 2018

@author: Sebastian
"""


import glob
from PIL import Image
import numpy as np
import os
from skimage import color
from skimage import io
from skimage.filters import threshold_mean

# process the images in the directories into gresycale and thresholded
def thresholdImages(path,threshold):
    Images = []
    thresholdedImages = []
    #labels = [i.split('\\')[-1][:-5] for i in files]
    for name in files: #for the image files in the directory
        #im = Image.open(name).convert('L')  #convert each image into greyscale  
        im = Image.open(name)
        #threshold_im = color.rgb2gray(io.imread('image.png')) #create thresholded image
        threshold_im = im
        Images.append(im) # append to arrays
        imageTuple = (threshold_im, name.split('\\')[-1][:-5])
        thresholdedImages.append(imageTuple)
    return Images, thresholdedImages

def normaliseImages(thresholdedImage_tuples):
    #for the thresholded images, we need to square them and resize them to 48x48
    thresholdedImages_resized = []
    for tuples in thresholdedImage_tuples:
        height,width = tuples[0].size
        if height != width: #image is not square
            print("\nHeight: {}, width: {}".format(height,width))
            resizingFactor = max(height,width)/48 #resize the larger dimension to 48
            if max(height,width) == height: #if the height is the larger variable
                print("Height is larger...")
                print("Initial dimensions {}".format(tuples[0].size))
                resized_im = tuples[0].resize((48,int(width/resizingFactor)),Image.ANTIALIAS) #keep aspect ratio
                new_im = Image.new("L", (48,48), (255))   ## produces a white 48x48 canvas
                new_im.paste(resized_im, (int((48-resized_im.size[0])/2),int((48-resized_im.size[1])/2)))
                print("Storing as {}".format(new_im.size))
                thresholdedImages_resized.append(new_im)
            else: #if width is the larger variable
                print("Width is larger...")
                print("Initial dimensions {}".format(tuples[0].size))
                resized_im = tuples[0].resize((int(height/resizingFactor),48),Image.ANTIALIAS) #keep aspect ratio
                new_im = Image.new("L", (48,48), (255))   ## produces a white 48x48 canvas
                new_im.paste(resized_im, (int((48-resized_im.size[0])/2),int((48-resized_im.size[1])/2)))
                print("Storing as {}".format(new_im.size))
                thresholdedImages_resized.append(new_im)
            #new_im.save(basePath+outputPath+'\\{}.png'.format(tuples[1]),'png')
        else: #if already square resize the images to 48x48 
            print("Height and width are equal...")
            print("Initial dimensions {}".format(tuples[0].size))
            resized_im = tuples[0].resize((48,48),Image.ANTIALIAS)
            new_im = Image.new("L", (48,48), (255))
            new_im.paste(resized_im, (0,0))
            print("Storing as {}".format(new_im.size))
            thresholdedImages_resized.append(new_im)
            
        final_im = new_im.point(lambda p: p > 180 and 255)  
        #final_im.show()
        final_im.save(basePath+outputPath+'\\{}.png'.format(tuples[1]),'png') 
    #images_as_arrays = np.asarray(thresholdedImages_resized)
    return thresholdedImages_resized#, images_as_arrays

def saveImages(Images): #saves the images in the output directory
    for i in range(len(Images)):
        Images[i].save(basePath+outputPath+'\\char_{}.png'.format(i),'png')

basePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Calligraphy\\'
inputPath = 'Individual_characters_calligraphy'
#inputPath = 'Segmented calligraphy'
outputPath = 'Individual_characters_greyscaled'
#outputPath = 'Normalised calligraphy'
path = basePath+inputPath+'\*'
files = glob.glob(path)
print(path)
threshold = 0
standardImages, thresholdedImage_tuples = thresholdImages(files,threshold) #greyscale, threshold
images = normaliseImages(thresholdedImage_tuples) #center and normalise the images
print("Saving images...")
#saveImages(images) #save them


