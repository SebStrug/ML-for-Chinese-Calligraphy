# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:55:15 2018

@author: Sebastian
"""

from InputTFRecord import inputs
import tensorflow as tf
import time
import math #to use radians in rotating the image
import random
from classDataManip import makeDir
import os
from PIL import Image
import numpy as np

#%% Initialisations
inputDim = 48
train_batch_size = 1
test_batch_size = 500
num_epochs = 1
num_output = 10
savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved runs\\'
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'

#%%
train_tfrecord_filename = localPath+'\\train'+str(num_output)+'.tfrecords'
test_tfrecord_filename = localPath+'\\test'+str(num_output)+'.tfrecords'

train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,\
                                              train_batch_size,num_epochs,normalize=False)
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,\
                                            test_batch_size,0,normalize=False)

with tf.Session() as sess:
    for i in range(10):
        print(train_image_batch.eval())
        print(train_image_batch.eval().shape)
        print(train_image_batch)
        saved_image = np.reshape(train_image_batch.eval()[0],(48,48))
        saved_label_onehot = tf.one_hot(train_label_batch,num_output).eval()[0]
        saved_label = train_label_batch.eval()
        
        print(saved_image)
        print(saved_label_onehot)
        print(saved_label)
        im = Image.fromarray(np.uint8((saved_image)*255))