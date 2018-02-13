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
import os, shutil
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
dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\gnt Files'

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

print("Number of unique characters: {}".format(len(list(set(train_labels)))))
print("Total number of samples: {}".format(len(train_labels)))

#%%Now we only want to save 10 unique characters
def generateUniqueAddrs(numUnique,trainType):
    """We are going to generate 10 unique characters in the addrs and labels"""
    print("Saving only {} unique characters".format(numUnique))
    startNum = 171
    if trainType == 'train':
        labels = train_labels
        addrs = train_addrs
    elif trainType == 'test':
        labels = test_labels
        addrs = test_addrs
    
    unique_labels = list(set(labels))[startNum:numUnique+startNum] #skip non-Chinese chars
    unique_addrs = []
    for i in unique_labels:
        tempString = '\\{}_copy'.format(i) #generate the string containing unique_label
        unique_addrs.append([i for i in addrs if tempString in i]) #find the addresses
    #unique_addrs is currently a list of lists, turn it into a flat list...
    unique_addrs = [item for sublist in unique_addrs for item in sublist]
    
    unique_labels = [addr[0:addr.index('_')] for addr in unique_addrs]
    unique_labels = [int(x.replace(saveImagePath+"\\train\\","")) for x in unique_labels]
    return unique_addrs, unique_labels

unique_train_addrs, unique_train_labels = generateUniqueAddrs(10,'train')
unique_test_addrs, unique_test_labels = generateUniqueAddrs(10,'test')
print("Number of unique characters: {}".format(len(list(set(unique_train_labels)))))
print("Total number of samples in training: {}".format(len(unique_train_labels)))
print("Total number of samples in testing: {}".format(len(unique_test_labels)))

#%%
# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generateTrainTFRecord(addrs,labels):
    print("Generating train TFRecord... \n")
    train_filename = 'train.tfrecords'
    # Initiating the writer and creating the train tfrecords file.
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(addrs)):
        # Load the image
        img = Image.open(addrs[i])
        img = np.array(img)
        label = labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()

def generateTestTFRecord(addrs,labels):
    print("Generating test TFRecord")
    test_filename = 'test.tfrecords'
    # Initiating the writer and creating the test tfrecords file.
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

generateTrainTFRecord(unique_train_addrs,unique_train_labels)
generateTestTFRecord(unique_test_addrs,unique_test_labels)

#%% Delete the saved images so they don't take up space
for the_file in os.listdir(saveImagePath):
    file_path = os.path.join(saveImagePath, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # the following line also removes sub-directories
        elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)