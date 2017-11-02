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

#get info on gnt file
data,tot = fF.iterateOverFiles(dataPathSeb)
dataInfo = fF.infoGNT(data,tot)
dataForSaving = fF.arraysFromGNT(data,dataInfo)
data=0;#delete data in raw byte form 

#create a labeled list of the unique chinese characters
#chinese characters are saved as numbers through ord function
labeledList = list(enumerate([ord(i) for i in set(dataForSaving[0])]))
print('Labels: saveIndex, saveChars')
#first array is the indices of the unique characters
#second index is each corresponding unique character
fF.saveNPZ(savePathSeb,'{}-zipped-list'.format(len(set(dataForSaving[0]))), \
           saveIndex=[i[0] for i in labeledList], \
           saveChars=[i[1] for i in labeledList])