# -*- coding: utf-8 -*-
"""
"""
#%%Load Data
#file Path for functions

user = "Seb"
#user = "Elliot"

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/forConversion/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'

if user == "Elliot":
    funcPath = funcPathElliot
    dataPath = dataPathElliot
    savePath = savePathElliot
elif user == "Seb":
    funcPath = funcPathSeb
    dataPath = dataPathSeb
    savePath = savePathSeb

#%%
import os
import numpy as np
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF
from classImageFunctions import imageFunc as iF
from PIL import Image
import warnings
import glob
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Extract data
#file path for data
#dataPath = 'C:\\Users\\ellio\\Documents\\training data\\forConversion'
dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\1.0 test'

#%%
#create a list of labels
os.chdir("C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0")
characters, GBCode, unicode = [],[],[];
with open("Char4037-list.txt", 'r') as file:
    file.readline()
    for line in file:
        characters.append(line.split()[0])
        GBCode.append(line.split()[1])
        unicode.append(line.split()[1])
        
#%%Check if the .gnt file is supposed to be training or test
with open("testlist.txt") as file:
    testGNT = [int(line.split()[0]) for line in file]

with open("trainlist.txt") as file:
    trainGNT = [int(line.split()[0]) for line in file]

#%%
def saveImages(saveImagePath,dataForSaving):
    #intialise the enumerated list
    enumerateList = []
    
    for i in range(len(dataForSaving[0])):
        singleChar = dataForSaving[0][i]
        singleImage = dataForSaving[1][i]
        
        if singleChar not in enumerateList:
            enumerateList.append(singleChar)
            
        dimension = int(singleImage.shape[0]**0.5)
        singleImage = Image.fromarray(np.resize(singleImage,(dimension,dimension)), 'L')
        
        copyVal = 0
        while os.path.exists('{}\\{}_copy{}.png'.format(saveImagePath,\
                             enumerateList.index(singleChar),copyVal)):
            copyVal += 1
        
        singleImage.save('{}\\{}_copy{}.png'.format(saveImagePath,\
                         enumerateList.index(singleChar),copyVal))
        
#%%
saveImagePath = "C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\savedImages"
gntPath = dataPath
imageSize = 48 #size of characters

def processGNTasImage(saveImagePath,gntPath,imageSize):
    totalFiles = 0
    for subdir, dirs, filenames in os.walk(gntPath):
        totalFiles += len(filenames)
    print("{} .gnt files".format(totalFiles))
    
    #create train and test folders
    if not os.path.exists(saveImagePath+"\\train"):
        os.mkdir(saveImagePath+"\\train")
    if not os.path.exists(saveImagePath+"\\test"):
        os.mkdir(saveImagePath+"\\test")
    
    for subdir, dirs, filenames in os.walk(gntPath):
        for file in filenames:
            print(file) #print the filename
            fullpath = os.path.join(subdir, file)
            with open(fullpath, 'rb') as openFile:
                byteInfo = fF.readByte(fullpath) #read in file as bytes
                openFile.close()
            dataInfo = fF.infoGNT(byteInfo) #get the info for one file
            #extract characters, images
            dataForSaving = fF.arraysFromGNT(byteInfo,dataInfo,imageSize)
            del byteInfo; del dataInfo
            if int(file[:file.index('-f.gnt')]) in trainGNT:
                saveImages(saveImagePath+"\\train",dataForSaving)
            elif int(file[:file.index('-f.gnt')]) in testGNT:
                saveImages(saveImagePath+"\\test",dataForSaving)
            else:
                print("Error, file name not in train or test")
                
processGNTasImage(saveImagePath,gntPath,imageSize)    

#%%
#Now store these as a TFRecord
#Start with the train labels

#read directories of each file
train_addrs = glob.glob(saveImagePath+"\\train\\*.png")
#label each file, first remove the '_copy0'
train_labels = [addr[0:addr.index('_')] for addr in train_addrs]
#then remove the first part
train_labels = [int(x.replace(saveImagePath+"\\train\\","")) for x in train_labels]
#checked and this contains all numbers in the range 0 - 4030 even if they are mislabeled here

#now repeat for the test files
test_addrs = glob.glob(saveImagePath+"\\test\\*.png")
test_labels = [addr[0:addr.index('_')] for addr in test_addrs]
test_labels = [int(x.replace(saveImagePath+"\\test\\","")) for x in test_labels]

# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

print("Generating train TFRecord")
train_filename = 'train.tfrecords'
# Initiating the writer and creating the tfrecords file.
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # Load the image
    img = Image.open(train_addrs[i])
    img = np.array(img)
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()

print("Generating test TFRecord")
test_filename = 'test.tfrecords'
# Initiating the writer and creating the tfrecords file.
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # Load the image
    img = Image.open(test_addrs[i])
    img = np.array(img)
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.test.Example(features=tf.test.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
