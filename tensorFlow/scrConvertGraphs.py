# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:27:13 2017

@author: Sebastian
"""
import os
import csv
import matplotlib.pyplot as plt
import glob
import numpy as np

savePathSeb = "C:/Users/Sebastian/Desktop/MLChinese/Saved script files/Saved graphs"

savePath = savePathSeb

#%% basic functions

def convWallTime(seconds):
    """Converts wall time from seconds to hours, minute, seconds. Day not saved"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h,24)
    h, m, s = int(h), int(m), int(s)
    return h,m,s

def findPlateau(values,limit):
    """Look for a plateau in accuracy,
       limit is in terms of a percentage e.g. 0.05 = 5%"""
    for i in range(1,len(values)):
        #if there is less than a 1% change in accuracy value
        changeInVal = (values[i]-values[i-1])/values[i]
        #print(values[i],values[i-1],changeInVal)
        if changeInVal < limit:
            break
    return values[i]

#%% manipulate data

def convertData(accData):
    """Converts data from strings into usable things, plus wall time"""
    #Get info from accuracy
    headers = accData[0]
    wallTime = [float(i[0]) for i in accData[1:]] 
    wallTime = [convWallTime(i) for i in wallTime]  
    stepTime = [int(i[1])/(10**4) for i in accData[1:]]
    accValues = [float(i[2][0:5]) for i in accData[1:]]
    
#    #Get values from xent
#    xentValues = [float(i[2][0:5]) for i in xentData[1:]]
#    minXent = min(xentValues); maxXent = max(xentValues)
#    #normalise the xent values
#    normXent = [(i - minXent)/(maxXent - minXent) for i in xentValues]
    return wallTime,stepTime,accValues#,normXent


#%% plotting functions

def plotSingle(stepTime,value,name):
    """Plots a single value for step time, e.g. name = "Accuracy" """
    lines = plt.plot(stepTime,value)
    plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0)
    plt.grid(True)
    plt.xlabel('Step time')
    plt.ylabel(name)
    plt.setp(lines, color='darkorange', linewidth=2.0)

def plotAccXent(stepTime,accValues,normXent):
    """Plots accuracy and xent along stepTime"""    
    accPlateau = findPlateau(accValues,0.01) #0.5% limit for plateau
    stepTimePlateau = stepTime[accValues.index(accPlateau)]
    plt.annotate('Accuracy: {}'.format(accPlateau), \
                 xy=(stepTimePlateau, accPlateau), \
                 xytext=(stepTimePlateau+30,accPlateau-0.3), \
                 horizontalalignment='right', verticalalignment='top',\
                 arrowprops=dict(width = 1, facecolor='gray', shrink=0.01))
    accuracy = plt.plot(stepTime,accValues)
    xent = plt.plot(stepTime,xentValues)
    plt.gca().set_color_cycle(['darkorange','slateblue'])
#    plt.legend([accuracy, xent], ['Accuracy','Cross entropy'])
#    plt.ylim(ymin=0,ymax=1)
#    plt.xlim(xmin=0)
    plt.grid(True)
    plt.xlabel('Step time')
    plt.ylabel('Value')
    plt.show()

#%% load in files
"""tensorboard csv to graph function"""
#change path so figure saves correctly
nameFile = "/fc1_out10_batch1024_LR"
os.chdir(savePath+nameFile)

def loadFiles(savePath,nameFile):
    files = glob.glob(savePath+nameFile+"\*.csv")
    allAcc = []
    for name in files:
        with open(name) as csvfile:
            accOneFile = [row for row in csv.reader(csvfile, delimiter=',')]
        allAcc.append(accOneFile)
    return allAcc

accData = loadFiles(savePath,nameFile)
wallTime = []; stepTime = []; accValues = []
for i in accData:
    singleWallTime,singleStepTime,singleAccValues = convertData(i)
    #delete every second value
    deleteNum = 10
    del singleStepTime[::deleteNum]; del singleWallTime[::deleteNum]; del singleAccValues[::deleteNum]
    wallTime.append(singleWallTime)
    stepTime.append(singleStepTime)
    accValues.append(singleAccValues)

batchSize = 1024
numOutputs = 50
trainRatio = 0.8
epochLength = (300*trainRatio*numOutputs)/batchSize

def smooth(y, box_pts):
    """Smooths with a moving average box (convolution)"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotAll(wallTime,stepTime,accValues):
    plt.figure()
    for i in range(len(accValues)):
        #plot charts
#        colours = ['salmon','royal blue','pale green','peach','grey',\
#                   'orange','dark read','khaki','seafoam','goldenrod']
        lines = plt.plot(stepTime[i],smooth(accValues[i],3))#, colours[i])
        plt.ylim(ymin=0,ymax=1)
        plt.xlim(xmin=0)
        plt.grid(True)
        plt.title('No. of outputs: {}, Batch size: {}, training ratio: {}, Epoch length: {}'.\
                  format(numOutputs,batchSize,trainRatio,int(epochLength)))
        plt.xlabel('Iterations *10^4')
        plt.ylabel('Accuracy')
        plt.setp(lines, linewidth=2.0)
        plt.savefig("Figure.svg")

plotAll(wallTime,stepTime,accValues)
