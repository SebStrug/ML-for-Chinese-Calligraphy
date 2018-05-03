"""
Script to convert gnt files to TFRecord files
"""

#%%
import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np

gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
#os.chdir(os.path.join(gitHubRep,"dataHandling/")) # << doesn't work for Seb
os.chdir('C:/Users/Sebastian/Desktop/GitHub/ML-for-Chinese-Calligraphy/dataHandling')

from classConvertToTFRecord import convertToTFRecord as cTF

#%% Set paths and the image size desired
imageSize = 48 #size of characters desired

# Main path containing gnt files, folders, and so on.
mainPath = "C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0"
#mainPath = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'

# Path containing gnt files
dataPath = mainPath + '\\gnt Files'
#dataPath = mainPath + '\\forConversion'
gntPath = dataPath

# Path to save the images to
saveImagePath = mainPath + "\\savedImages"

#Choose to generate TFRecords for a GAN or Neural net
what_net = 'Neural'
#what_net = 'GAN'

#%% Process the calligraphy characters into tfrecords files
imagePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Calligraphy\\Individual_characters_greyscaled'
addrs = glob.glob(imagePath+'\*')
tuples = [(i,int(i.split('\\image')[-1][:-4])) for i in addrs]
print("Number of calligraphy characters: {}".format(len(addrs)))
filename = 'calligraphy_greyscaled.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)
for i in range(len(tuples)):
    # Load the image
    img = Image.open(tuples[i][0])
    img = np.array(img)
    label = tuples[i][1]
    # Create a feature
    feature = {'train/label': cTF._int64_feature(label),
               'train/image': cTF._bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()   

#%% Process the intersection of the calligraphy and CASIA into a tfrecords file
imagePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\savedImages\\test'
addrs = glob.glob(imagePath+'\*')
def convert_addrs(addrs):
    return addrs.split('test\\')[1].split('_copy')[0]

filename_output = 'calligraphy_CASIA.tfrecords'
writer = tf.python_io.TFRecordWriter(filename_output)

tfrecord_addrs = [i for i in addrs if int(convert_addrs(i)) in desiredLabels]
tfrecord_labels = [int(convert_addrs(i)) for i in tfrecord_addrs]
for i in range(len(tfrecord_addrs)):
    # Load the image
    img = Image.open(tfrecord_addrs[i])
    img = np.array(img)
    label = tfrecord_labels[i]
    # Create a feature
    feature = {'test/label': cTF._int64_feature(label),
               'test/image': cTF._bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()

#%%Check if the .gnt file is supposed to be training or test
def checkTrainTest():
    os.chdir(mainPath)
    with open("testlist.txt") as file:
        testGNT = [int(line.split()[0]) for line in file]
    with open("trainlist.txt") as file:
        trainGNT = [int(line.split()[0]) for line in file]
    return testGNT, trainGNT

#read directories of each image in training, and get their corresponding labels
def read_dir(saveImagePath):
    train_addrs = glob.glob(saveImagePath+"\\train\\*.png")
    train_labels = cTF.convLabels(saveImagePath,'train',train_addrs)
    #now repeat for the test files
    test_addrs = glob.glob(saveImagePath+"\\test\\*.png")
    test_labels = cTF.convLabels(saveImagePath,'test',test_addrs)
    #now a combination of train and test for the .gnt files
    generic_addrs = glob.glob(saveImagePath+"\\generic\\*.png")
    generic_labels = cTF.convLabels(saveImagePath,'generic',generic_addrs)
    #store as single array for readability
    addrs_labels = [train_addrs, train_labels, test_addrs, test_labels, generic_addrs, generic_labels] 

    numUniq = 0; numTotal = 0;
    for i in addrs_labels:
        numUniq += len(list(set(i)) + list(set(i)))
        numTotal += len(i) + len(i)
    print("Number of unique characters in training and test: {}".format(numUniq))
    print("Total number of samples in training and test: {}".format(numTotal))
    print("Average number of samples per character: {}\n".format(numTotal/numUniq))
    return addrs_labels

#create a list of gnt files that will be split into training or test
testGNT,trainGNT = checkTrainTest()

#process the images from the .gnt files and save them (as images)
#this is the majority of the processing
if what_net == 'Neural':
    #for the neural nets/supervised learning
    print("Processing for a neural net")
    cTF.processGNTasImage(saveImagePath,gntPath,imageSize,trainGNT,testGNT)
    # Saves training and test files as tfrecords separately, for supervised learning
    #save TFRecord files containing the following numbers of unique Chinese characters
    for numOutputs in [10]:
        #generate the addresses and corresponding labels of all Chinese characters 
        #that are in the set of X unique Chinese characters being saved
        addrs_labels = read_dir(saveImagePath)
        unique_train_addrs, unique_train_labels = \
            cTF.generateUniqueAddrs(saveImagePath,numOutputs,'train',addrs_labels)
        unique_test_addrs, unique_test_labels = \
                cTF.generateUniqueAddrs(saveImagePath,numOutputs,'test',addrs_labels)
        os.chdir(mainPath)
        cTF.generateTrainTFRecord(unique_train_addrs,unique_train_labels,numOutputs)
        cTF.generateTestTFRecord(unique_test_addrs,unique_test_labels,numOutputs)
    
elif what_net == 'GAN':
    # for the GAN
    print("Processing for a GAN")
    cTF.processGNTasImageGeneric(saveImagePath,gntPath,imageSize,trainGNT,testGNT) 
    # This is for processing the images into a generic tfrecords file, that lumps together training and test
    # this is for the GAN
    for numOutputs in [3866]:
        #generate the addresses and corresponding labels of all Chinese characters 
        #that are in the set of X unique Chinese characters being saved
        addrs_labels = read_dir(saveImagePath)
        unique_addrs, unique_labels = cTF.generateUniqueAddrs(saveImagePath,numOutputs,'generic',addrs_labels)
        os.chdir(mainPath)
        cTF.generateGenericTFRecord(unique_addrs,unique_labels,numOutputs)       

else: 
    print("Did not choose Neural or GAN, exiting")
    
#%%
#delete the saved images
#cTF.delete_images(saveImagePath)