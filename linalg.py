# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:07:19 2017

@author: Sebastian
"""

import numpy as np
import numpy.linalg as lin

matrix = [[1,2,3],[2,4,1],[3,6,5]]
matrix2 = lin.inv(matrix)
print(matrix2)