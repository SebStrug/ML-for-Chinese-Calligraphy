# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:38:20 2017

@author: Sebastian
"""
import numpy as np
from PIL import Image
from skimage.measure import block_reduce
import scipy.misc

path = 'C:/Users/Sebastian/Desktop/MLChinese/sample 46.png'

class imageFunc(object):
    """Class of functions that perform various things on images"""
    def extendArray(array, maxHeight, maxWidth):
        """Generates a new array of zeros with a size defined by the max height and max width,
        with the original array in question in the centre of that array."""
        height = array.shape[0]
        width = array.shape[1]
        if width <= maxWidth and height <= maxHeight:
            newArray = np.full((maxHeight,maxWidth),255)
            lowerBound = maxHeight//2 - height//2
            upperBound = lowerBound+height
            leftBound = maxWidth//2 - width//2
            rightBound = leftBound + width
            newArray[lowerBound:upperBound, leftBound:rightBound] = array
            return newArray
        else:
            print("Error, max dimension(s) less than dimension(s) of array. \
                  Width = {}, Height = {}, Max width = {}, Max Height = {}". \
                  format(width,height,maxWidth,maxHeight))
            return array
        
    #Function to reduce an array to scale it down
    #reduceArray
    #something about "skimage using an aggregate approach" is important
    def downscaleArray(array,height,width):
        #this is fine as long as we are not binarizing the data
        #it takes the means of slices
        newArray = block_reduce(array, block_size=(height,width), func=np.mean) 
        #can also do np.max
        newArray = np.ceil(newArray).astype(int) #round up and return integers for each element
        return newArray
    
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
        whiteImage = imageFunc.extendArray(upscaledImage,newDimension,newDimension)
        #reduce the image to an appropriate size
        blockSize = int(newDimension/downscaleSize)
        reducedArray = imageFunc.downscaleArray(whiteImage,blockSize,blockSize)
        #show the image from the array with
        #img = Image.fromarray(upscaledArray).show()
        return reducedArray

    def binarizeArray(array,threshold):
        trueFalse = array < threshold
        #store white as 0, black as 1
        return trueFalse.astype(int)
    
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
        return np.array(image.getdata()).astype(np.int)