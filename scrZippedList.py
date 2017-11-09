# -*- coding: utf-8 -*-
"""
Creates a zipped list labeling each unique character in our data
Need to run this script for all the files we have to create a master zipped list
    with all unique values.
    Until then, this is a rough approximation returning 4052 unique characters.
"""
import os
import numpy as np

dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\HWDtest2'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'

os.chdir(funcPathSeb)
from classFileFunctions import fileFunc as fF

def dataFromGNT(dataPath):
    #get info on gnt file
    data,tot = fF.iterateOverFiles(dataPathSeb)
    dataInfo = fF.infoGNT(data,tot)
    dataForSaving = fF.arraysFromGNT(data,dataInfo)
    data=0;#delete data in raw byte form 
    return dataForSaving[0]
    #to streamline this script, we only need to read in the characters from all the files,
    #right now we are still reading in the images.

#don't need to read in NPZ files
#NPZPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
#def dataFromNPZ(dataPath):
#    labels1, images1 = fF.readNPZ(NPZPathSeb,'1001-1100C','saveLabels','saveImages')
#    labels2, images2 = fF.readNPZ(NPZPathSeb,'1101-1200C','saveLabels','saveImages')
#    labels3, images3 = fF.readNPZ(NPZPathSeb,'1201-1300C','saveLabels','saveImages')
#    del images1; del images2; del images3;
#    return np.concatenate((labels1,labels2,labels3),axis=0)

labels = dataFromGNT(dataPathSeb)
#create a labeled list of the unique chinese characters
#chinese characters are saved as numbers through ord function
labeledList = list(enumerate([ord(i) for i in set(labels)]))
print('Labels: saveIndex, saveChars')
#first array is the indices of the unique characters
#second index is each corresponding unique character
fF.saveNPZ(savePathSeb,'{}-zipped-list'.format(len(set(labels))), \
           saveIndex=[i[0] for i in labeledList], \
           saveChars=[i[1] for i in labeledList])

"""Now we need to read in files and write a function to find the 
    corresponding label to each sample (character) we have"""
index, chars = fF.readNPZ(savePathSeb,'{}-zipped-list'.format(len(set(labels))),'saveIndex','saveChars')
#create a list containing a label for each sample (character) we have
labeledChars = [np.where(chars==i)[0][0] for i in [ord(i) for i in labels] ]