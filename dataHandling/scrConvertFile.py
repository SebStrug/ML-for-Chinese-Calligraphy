"""
Script to convert gnt files to TFRecord files
"""

#%%
import os
import glob

gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))

from classConvertToTFRecord import convertToTFRecord as cTF

#%% Set paths and the image size desired
imageSize = 48 #size of characters desired

# Main path containing gnt files, folders, and so on.
mainPath = "C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0"
#dataPath = 'C:\\Users\\ellio\\Documents\\training data'

# Path containing gnt files
dataPath = mainPath + '\\1.0 test'
#dataPath = mainPath + '\\forConversion'
gntPath = dataPath

# Path to save the images to
saveImagePath = mainPath + "\\savedImages"

#%%Check if the .gnt file is supposed to be training or test
def checkTrainTest():
    os.chdir(mainPath)
    with open("testlist.txt") as file:
        testGNT = [int(line.split()[0]) for line in file]
    with open("trainlist.txt") as file:
        trainGNT = [int(line.split()[0]) for line in file]
    return testGNT, trainGNT

#create a list of gnt files that will be split into training or test
testGNT,trainGNT = checkTrainTest()

#process the images from the .gnt files and save them (as images)
#this is the majority of the processing
cTF.processGNTasImage(saveImagePath,gntPath,imageSize,trainGNT,testGNT)                

#%%
#read directories of each image in training, and get their corresponding labels
train_addrs = glob.glob(saveImagePath+"\\train\\*.png")
train_labels = cTF.convLabels(saveImagePath,'train',train_addrs)
#now repeat for the test files
test_addrs = glob.glob(saveImagePath+"\\test\\*.png")
test_labels = cTF.convLabels(saveImagePath,'test',test_addrs)
addrs_labels = [train_addrs,train_labels,test_addrs,test_labels] #store as single array for readability

print("Number of unique characters: {}".format(len(list(set(train_labels)))))
print("Total number of samples: {}".format(len(train_labels)))

#save TFRecord files containing the following numbers of unique Chinese characters
for numOutputs in [10,20,30,50,100]:
    #generate the addresses and corresponding labels of all Chinese characters 
    #that are in the set of X unique Chinese characters being saved
    unique_train_addrs, unique_train_labels = \
        cTF.generateUniqueAddrs(saveImagePath,numOutputs,'train',addrs_labels)
    unique_test_addrs, unique_test_labels = \
            cTF.generateUniqueAddrs(saveImagePath,numOutputs,'test',addrs_labels)
    cTF.generateTrainTFRecord(unique_train_addrs,unique_train_labels,numOutputs)
    cTF.generateTestTFRecord(unique_test_addrs,unique_test_labels,numOutputs)

#delete the saved images
#cTF.delete_images(saveImagePath)