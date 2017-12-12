# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:27:13 2017

@author: Sebastian
"""
import os
import csv
import matplotlib.pyplot as plt

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
#tensorboard csv to graph function
savePath = "C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files"
os.chdir(savePath)
#Read in accuracy and xent
accuracy = "ExampleAccuracyCSV.csv"
xent = "ExampleXENTCSV.csv"

def loadFiles(accFileName,xentFileName):
    with open(accuracy) as csvfile:
        accData = [row for row in csv.reader(csvfile, delimiter=',')]
    with open(xent) as csvfile:
        xentData = [row for row in csv.reader(csvfile, delimiter=',')]
    return accData, xentData
    
#%% manipulate data

def manipulateData(accData,xentData):
    #Get info from accuracy
    headers = accData[0]
    wallTime = [float(i[0]) for i in accData[1:]] 
    wallTime = [convWallTime(i) for i in wallTime]  
    stepTime = [int(i[1]) for i in accData[1:]]
    accValues = [float(i[2][0:5]) for i in accData[1:]]
    
    #Get values from xent
    xentValues = [float(i[2][0:5]) for i in xentData[1:]]
    minXent = min(xentValues); maxXent = max(xentValues)
    #normalise the xent values
    normXent = [(i - minXent)/(maxXent - minXent) for i in xentValues]
    return wallTime,stepTime,accValues,normXent





