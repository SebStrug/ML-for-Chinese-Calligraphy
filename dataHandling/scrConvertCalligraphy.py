# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:25:38 2018

@author: Sebastian
"""


import glob
from PIL import Image
import numpy as np
import os

# process the images in the directories into gresycale and thresholded
def thresholdImages(path,threshold):
    Images = []
    thresholdedImages = []
    for name in files: #for the image files in the directory
        im = Image.open(name).convert('L')  #convert each image into greyscale  
        threshold_im = im.point(lambda p: p > threshold and 255) #create thresholded image
        Images.append(im) # append to arrays
        thresholdedImages.append(threshold_im)
    return Images, thresholdedImages

def normaliseImages(thresholdedImages):
    #for the thresholded images, we need to square them and resize them to 48x48
    thresholdedImages_resized = []
    for image in thresholdedImages:
        height,width = image.size
        if height != width: #image is not square
            print("\nHeight: {}, width: {}".format(height,width))
            resizingFactor = max(height,width)/48 #resize the larger dimension to 48
            if max(height,width) == height: #if the height is the larger variable
                print("Height is larger...")
                print("Initial dimensions {}".format(image.size))
                resized_im = image.resize((48,int(width/resizingFactor)),Image.ANTIALIAS) #keep aspect ratio
                new_im = Image.new("1", (48,48), (255))   ## produces a white 48x48 canvas
                new_im.paste(resized_im, (int((48-resized_im.size[0])/2),int((48-resized_im.size[1])/2)))
                print("Storing as {}".format(new_im.size))
                thresholdedImages_resized.append(new_im)
            else: #if width is the larger variable
                print("Width is larger...")
                print("Initial dimensions {}".format(image.size))
                resized_im = image.resize((int(height/resizingFactor),48),Image.ANTIALIAS) #keep aspect ratio
                new_im = Image.new("1", (48,48), (255))   ## produces a white 48x48 canvas
                new_im.paste(resized_im, (int((48-resized_im.size[0])/2),int((48-resized_im.size[1])/2)))
                print("Storing as {}".format(new_im.size))
                thresholdedImages_resized.append(new_im)
        else: #if already square resize the images to 48x48 
            print("Height and width are equal...")
            print("Initial dimensions {}".format(image.size))
            image = image.resize((48,48),Image.ANTIALIAS)
            print("Storing as {}".format(image.size))
            thresholdedImages_resized.append(image)
    #images_as_arrays = np.asarray(thresholdedImages_resized)
    return thresholdedImages_resized#, images_as_arrays

def saveImages(Images): #saves the images in the output directory
    for i in range(len(Images)):
        Images[i].save(basePath+outputPath+'\\char_{}.png'.format(i),'png')

basePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Calligraphy\\'
inputPath = 'Individual_characters_calligraphy'
#inputPath = 'Segmented calligraphy'
outputPath = 'Individual_characters_normalised'
#outputPath = 'Normalised calligraphy'
path = basePath+inputPath+'\*'
files = glob.glob(path)
print(path)
threshold = 160
standardImages, thresholdedImages = thresholdImages(files,threshold) #greyscale, threshold
images = normaliseImages(thresholdedImages) #center and normalise the images
saveImages(images) #save them


