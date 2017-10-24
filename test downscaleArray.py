# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:02:00 2017

@author: Sebastian
"""

#Function to reduce an array to scale it down
#reduceArray
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)