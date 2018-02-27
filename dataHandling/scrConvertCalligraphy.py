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
        else:
            image = image.resize(int(height/resizingFactor,48),Image.ANTIALIAS) #keep aspect ratio
    else: #if already square resize the images to 48x48 
        image = image.resize((48,48),Image.ANTIALIAS)
    thresholdedImages_resized.append(image)
    
def scale(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio 
    and place it in center of white 'max_size' image 
    """
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)), method)
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]), method)
 
    offset = (((max_size[0] - scaled.size[0]) / 2), ((max_size[1] - scaled.size[1]) / 2))
    back = Image.new("RGB", max_size, "white")
    back.paste(scaled, offset)
    return back


