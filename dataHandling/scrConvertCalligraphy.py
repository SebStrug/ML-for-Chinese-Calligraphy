# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:25:38 2018

@author: Sebastian
"""


import glob
from PIL import Image

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
            resizingFactor = max(height,width)/48 #resize the larger dimension to 48
            if max(height,width) == height: #if the height is the larger variable
                image = image.resize((48,int(width/resizingFactor)),Image.ANTIALIAS) #keep aspect ratio
                new_im = Image.new("1", (48,48), (255))   ## produces a white 48x48 canvas
                new_im.paste(image, (int((48-image.size[0])/2),int((48-image.size[1])/2)))
                thresholdedImages_resized.append(new_im)
            else: #if width is the larger variable
                image = image.resize(int(height/resizingFactor,48),Image.ANTIALIAS) #keep aspect ratio
                thresholdedImages_resized.append(image)
        else: #if already square resize the images to 48x48 
            image = image.resize((48,48),Image.ANTIALIAS)
            thresholdedImages_resized.append(image)
    return thresholdedImages_resized

def saveImages(Images): #saves the images in the output directory
    for i in range(len(Images)):
        Images[i].save(basePath+outputPath+'\\char_{}.png'.format(i),'png')

basePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Calligraphy\\'
inputPath = 'Segmented calligraphy'
outputPath = 'Normalised calligraphy'
files = glob.glob(basePath + inputPath+"\*.jpg")
threshold = 80 
standardImages, thresholdedImages = thresholdImages(files,threshold) #greyscale, threshold
Images = normaliseImages(thresholdedImages) #center and normalise the images
saveImages(Images) #save them


