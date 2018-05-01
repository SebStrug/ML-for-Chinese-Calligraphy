# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:27:13 2017

@author: Sebastian
"""
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import numpy as np
from datetime import datetime
from datetime import timedelta

gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
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

def convertData(accData):
    """Converts data from strings into usable things, plus wall time"""
    wallTime = []; stepTime = []; accValues = [];
    relativeTime = [];
    for i in range(len(accData)):
        print(i)
        starting_wallTime = datetime.fromtimestamp(float(accData[i][1][0]))
        wallTime_one = [float(j[0]) for j in accData[i][1:]] # get time in seconds
        wallTime_one = [datetime.fromtimestamp(i) for i in wallTime_one]
        #wallTime_one = [j - wallTime_one[0] for j in wallTime_one] #get relative time
        
        relativeTime_one = [i - starting_wallTime for i in wallTime_one]
        relativeTime_mins = [i.seconds/60 for i in relativeTime_one]
        
        #wallTime_one = [matplotlib.dates.date2num(i) for i in wallTime_one]
#        relativeTime_one = [convWallTime(j) for j in wallTime_one]  # convert to hours, mins, seconds
        stepTime_one = [int(j[1])/(10**3) for j in accData[i][1:]] # get the step count
        accValues_one = [float(j[2][0:5]) for j in accData[i][1:]] #get the accuracy values
        
        relativeTime.append(relativeTime_mins)
        wallTime.append(wallTime_one)
        stepTime.append(stepTime_one)
        accValues.append(accValues_one)
    return wallTime, stepTime, accValues, relativeTime

def smooth(y, box_pts):
    """Smooths with a moving average box (convolution)"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def loadFiles(savePath,nameFile):
    files = glob.glob(savePath+nameFile+"\*.csv")
    allAcc = []
    for name in files:
        with open(name) as csvfile:
            accOneFile = [row for row in csv.reader(csvfile, delimiter=',')]
        allAcc.append(accOneFile)
    return allAcc

def plotAll(wallTime, relTime, stepTime, accValues, plot_type='step', smoothVal = 1):
    plt.figure()
    for i in range(len(accValues)):
        if plot_type == 'step': #if we want to plot the step count
            lines = plt.plot(stepTime[i],smooth(accValues[i],smoothVal))
            plt.ylim(ymin=0,ymax=1)
            plt.xlim(xmin=0,xmax=9)
            plt.xlabel('Iterations *10^3')
            plt.title('No. of outputs: {}, Batch size: {}'.\
              format(numOutputs,batchSize))
            plt.grid(True)
            plt.ylabel('Test accuracy')
            plt.setp(lines, linewidth=2.0)
            plt.savefig("Figure_step.svg")
            plt.savefig("Figure_step.png")
        elif plot_type == 'relative':
            lines = plt.plot(relTime[i],smooth(accValues[i],smoothVal))
            plt.ylim(ymin=0,ymax=1)
            plt.xlim(xmin=0)
            plt.xlabel('Relative time in minutes')
            plt.title('No. of outputs: {}, Batch size: {}'.\
              format(numOutputs,batchSize))
            plt.grid(True)
            plt.ylabel('Test accuracy')
            plt.setp(lines, linewidth=2.0)
            plt.savefig("Figure_relative.svg")
            plt.savefig("Figure_relative.png")            
        elif plot_type == 'wall':
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            lines = plt.plot(wallTime[i],smooth(accValues[i],smoothVal))
            plt.gcf().autofmt_xdate()
            plt.xlabel('Time in hours:minutes')
            plt.title('No. of outputs: {}, Batch size: {}'.\
              format(numOutputs,batchSize))
            plt.grid(True)
            plt.ylabel('Test accuracy')
            plt.setp(lines, linewidth=2.0)
            plt.savefig("Figure_wall.svg")
            plt.savefig("Figure_wall.png")
        else:
            print('Bad input entered')
            break
        


#%% load in files
"""tensorboard csv to graph function"""
#change path so figure saves correctly
dataPath, LOGDIR, rawDataPAth = fF.whichUser("Seb")


savePathElliot = LOGDIR
savePathSeb = "C:/Users/Sebastian/Desktop/MLChinese/Saved_runs/"

savePath = savePathSeb
nameFileElliot = "CSV"
nameFileSeb = '/2conv100_comparePooling'
nameFile = nameFileSeb

os.chdir(savePath+nameFile)

accData = loadFiles(savePath,nameFile)
wallTime, stepTime, accValues, relTime = convertData(accData)

batchSize = 128
numOutputs = 100

plotAll(wallTime, relTime, stepTime, accValues, plot_type = 'relative', smoothVal = 2)
