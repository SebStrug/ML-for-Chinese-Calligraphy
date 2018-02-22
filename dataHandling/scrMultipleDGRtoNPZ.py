# -*- coding: utf-8 -*-
"""
Spyder Editor
#
This is a temporary script file.
"""
#%%Load Data
#file Path for functions

#user = "Seb"
user = "Seb"

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

#%% Extract data
#file path for data
#dataPath = 'C:\\Users\\ellio\\Documents\\training data\\forConversion'
dataPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
numFiles = len([filenames for subdir, dirs, filenames in os.walk(dataPath)][0])
    
#get info on gnt file
data,tot = fF.iterateOverFiles(dataPath)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo) #create images that are 48x48
del data #delete data in raw byte form 

#%% Check our characters and their labels are lining up in the original data file
characters = dataForSaving[0]
images = dataForSaving[1]
#del dataForSaving #clear memory
for i in range(len(characters)):
    if characters[i] == characters[0]:
        print(i,characters[i])
    elif characters[i] == characters[1]:
        print(i,characters[i])

#%% Create a zipped list, labeling each characters
def createZipList(characters):
    #each of the characters in this list has a unique index
    fF.saveNPZ(savePath,"3755charsZipped",saveChars = list(set(characters)))

#%% Now label each file according to the zipped list
#open the file with uniquely indexed characters
with open(os.path.join(savePath,"3755charsZipped"), 'rb') as ifs:
    fileNPZ = np.load(ifs)
    uniqueChars = fileNPZ["saveChars"]

#label our original list of characters
labeledChars = [np.where(uniqueChars == i)[0][0] for i in characters]

#save a file containing our images (as arrays) and corresponding labels
fF.saveNPZ(savePath,"CharToNumList_{}".format(numFiles),\
           saveImages = dataForSaving[1], \
           saveLabels = labeledChars)

#%% Check that the images and labels match up
#savePath = C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted
def checkImages(savePath,nameCharToNumList,charNumToCheck):
    from PIL import Image
    
    with open(os.path.join(savePath,nameCharToNumList), 'rb') as ifs:
        fileNPZ = np.load(ifs)
        imagesCheck = fileNPZ["saveImages"]
        labelsCheck = fileNPZ["saveLabels"]
        
    allImg = []
    for i in range(len(imagesCheck)):
        if labelsCheck[i] == charNumToCheck:
            print(i,labelsCheck[i])
            allImg.append(Image.fromarray(np.resize(imagesCheck[i],(40,40)), 'L'))
    return allImg
        
            
