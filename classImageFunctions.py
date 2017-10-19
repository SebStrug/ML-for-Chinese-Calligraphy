# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:38:20 2017

@author: Sebastian
"""
import numpy as np
from PIL import Image

path = 'C:/Users/Sebastian/Desktop/MLChinese/sample 46.png'

class imageFunc(object):
    """Class of functions that perform various things on images"""
    def extendArray(array, width, height, maxWidth, maxHeight):
        """Generates a new array of zeros with a size defined by the max height and max width,
        with the original array in question in the centre of that array."""
        if width < maxWidth and height < maxHeight:
            newArray = np.zeros((maxHeight,maxWidth))
            lowerBound = maxHeight//2 - height//2
            upperBound = lowerBound+height
            leftBound = maxWidth//2 - width//2
            rightBound = leftBound + width
            newArray[lowerBound:upperBound, leftBound:rightBound] = array
        else:
            newArray = array
        return newArray
    
    def scaleImage(path,output):
        """Scales the image by a random value between 0.9 and 1.1"""
        im = Image.open(path) #not arary but image
        scaling = np.random.uniform(0.9,1.1) #scales by a random value between 0.9 and 1.1
        scaledSize = [i*scaling for i in im.size] #creates size array
        im.thumbnail(scaledSize, Image.ANTIALIAS)
        im.save(output)
        
    def rotateImage(path,output):
        """Rotates the image by a random value (in degrees) between -10 and 10"""
        im = Image.open(path)
        rotation = np.random.uniform(-10,10)
        im = im.rotate(rotation)
        im.save(output)
    
    def translateImage(path,output):
        """Translates the image by a random value between -10 and 10 in the x and y directions"""
        img = Image.open(path)
        xyTranslate = np.random.uniform(-10,10,2)
        a = 1; b = 0;
        c = xyTranslate[0] #left/right
        d = 0; e = 1;
        f = xyTranslate[1] #up/down
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        img.save(output)
        print(xyTranslate)
        
    def reduceImage(path,output,pixelSize):
        """Reduces image to a set size specified by pixelSize"""
        """Suggested: 40"""
        im = Image.open(path)
        imSize = im.size
        factor = imSize[0]/pixelSize 
        newSize = (pixelSize,int(imSize[1]/factor))
        # I downsize the image with an ANTIALIAS filter (gives the highest quality)
        im = im.resize(newSize,Image.ANTIALIAS)
        im.save(output,optimize=True,quality=95)
        
    def arrayToImage(array,height,width):
        """Generates image from an array"""
        #change from array to matrix OR checks it's the right size
        np.reshape(array,(height,width),'C') 
        #print(array.shape,'\n')
        img = Image.fromarray(array); #generates image
        return img;

    def resizeImage(image,newWidth,newHeight ):
        """Resizes an image"""
        newim=Image.new("L",(newWidth,newHeight)) #create new white canvas
        #Paste old image, centred onto new canvas
        newim.paste(image,(int((newWidth-image.size[0])/2),int((newHeight-image.size[1])/2)))
        return newim
    
    def PIL2array(image):
        """Changes a PIL image to an array"""
        return np.array(image.getdata(), np.uint8)