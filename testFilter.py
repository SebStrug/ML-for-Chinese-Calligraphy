# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:39:44 2017

@author: Sebastian
"""

def f(x): 
    return x % 2 != 0 and x % 3 != 0


print(list(filter(f, range(2, 25))))