# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:25:38 2018

@author: Sebastian
"""


import glob
from PIL import Image

path = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Calligraphy'
files = glob.glob(path+"\*.jpg")
threshold = 80 

# process the images in the directories into gresycale and thresholded
Images = []
thresholdedImages = []
for name in files: #for the image files in the directory
    im = Image.open(name).convert('L')  #convert each image into greyscale  
    threshold_im = im.point(lambda p: p > threshold and 255) #create thresholded image
    Images.append(im) # append to arrays
    thresholdedImages.append(threshold_im)
    
#for the thresholded images, we need to square them and resize them to 48x48
#resize the larger dimension to 48
thresholdedImages_resized = []
for image in thresholdedImages:
    height,width = image.size
    if height != width: #image is not square
        resizingFactor = max(height,width)/48
        if max(height,width) == height:
            image = image.resize((48,int(width/resizingFactor)),Image.ANTIALIAS) #keep aspect ratio
            old_size = image.size
            new_im = Image.new("RGB", (48,48))   ## luckily, this is already black!
            new_im.paste(image, ((48-old_size[0])/2,(48-old_size[1])/2))
        else:
            image = image.resize(int(height/resizingFactor,48),Image.ANTIALIAS) #keep aspect ratio
    else: #if already square resize the images to 48x48 
        image = image.resize((48,48),Image.ANTIALIAS)
    thresholdedImages_resized.append(image)
    



