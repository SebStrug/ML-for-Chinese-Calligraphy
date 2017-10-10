# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import os
from PIL import Image

#file path for data
filePath = 'C:/Users/ellio/Desktop/training data/first download/HWDB1.1trn_gnt_P1/'

def conv(byte):
    return int.from_bytes(byte, byteorder="little")

def arrayToImage(array,height,width):
    np.reshape(array,(height,width),'C')#resize it
    print(array.shape,'\n')
    img = Image.fromarray(array);
    return img;



#set path and open file
os.chdir(filePath)
with open("1001-c.gnt", 'rb') as f:
    fullFile = f.readlines()[0];
    f.close();
#print (fullFile)
#%%calculate number of samples in the data and the max height and width
samples = 0;
position = 0;
totalSize = 0;
maxWidth=0;
maxHeight=0;
print (len(fullFile))
while position < len(fullFile):
    sampleSize = conv(fullFile[position:position+4]);
    maxWidth = max(conv(fullFile[position+6:position+8]),maxWidth)
    maxHeight = max(conv(fullFile[position+8:position+10]),maxHeight)
    samples+=1;
    position += sampleSize
    print(samples)
    print(sampleSize)
    totalSize +=sampleSize;
print('total samples:', samples ,'\n')
print('total size:', totalSize ,'\n')  
 #%%  
 #create arrays to store data read in
sampleSize = np.zeros(samples,np.uint32);
width = np.zeros(samples,np.uint16);
height = np.zeros(samples,np.uint16);
character = np.zeros(samples,np.unicode);
#place data into arrays
position = 0;
i = 0;

while i < samples-1:
    sampleSize[i] = conv(fullFile[position:position+4]);
    character[i] = fullFile[position+4:position+6].decode('gb2312');
    width[i] = conv(fullFile[position+6:position+8]);
    height[i] = conv(fullFile[position+8:position+10]);
    imageSize = width*height;
    image = np.zeros((height[i],width[i]))
    for j in range(0,height[i]):
        for k in range(0,width[i]):
            image[j][k]=fullFile[position+10+j*width[i]+k];
    position +=sampleSize[i];
    print(i,'\n')
    print('samplSize',sampleSize[i])
    print('character',character[i])
    print('width',width[i])
    print('height',height[i])
    im = arrayToImage(image,height[i],width[i])
    fileName = 'sample %(number)d.png'% {"number":i};
    newim=Image.new("RGB",(maxWidth,maxHeight),(255,255,255))
    newim.paste(im,(int((maxWidth-im.size[0])/2),int((maxHeight-im.size[1])/2)))
    newim.save(fileName);
    i+= 1;
