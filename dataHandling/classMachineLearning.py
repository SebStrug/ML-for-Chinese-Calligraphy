# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:24:04 2017

@author: Sebastian
"""
from classFileFunctions import fileFunc as fF
import numpy as np

class machineLearning(object):
    """Class of machine learning related functions"""    
    def storeCharNumber(character,name,path):
        listCharacters,index = fF.readNPZ(path,name,'saveChars','saveIndex')
        return np.where(listCharacters==ord(character))[0][0]
        
    def newHotOnes(characters,name,path):
        characters = [ord(i) for i in characters] #convert to numbers
        zippedList = machineLearning.createZippedList(characters,name,path)
        #generate list of all characters, then find the corresponding index number for each character
        charVals = [[i[0] for i in zippedList].index(val) for val in characters]
        hotOnes = np.zeros((len(characters),len(set(characters)))) #array of zeros (10*3755),3755
        hotOnes[np.arange(len(characters)), charVals] = 1
        hotOnes = [hotOnes[i].astype(int) for i in range(len(hotOnes))]
        return hotOnes
    