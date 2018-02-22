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

train_kwargs = {"normalize_images": False, "augment_images": False, "shuffle_data": True}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,\
                                              train_batch_size,num_epochs,\
                                              **train_kwargs)
test_kwargs = {"normalize_images": False, "augment_images": False, "shuffle_data": True}
test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,\
                                            test_batch_size,0,\
                                            **test_kwargs)

with tf.Session() as sess:
    for i in range(200):
        test_images, test_labels = sess.run([test_image_batch,test_label_batch])
        print("Print the sess.run")
        print(test_images)
        print(test_labels)
        print(tf.one_hot(test_labels,10).eval()[0])
        

        saved_image = np.reshape(test_images[0],(48,48))
        saved_label_onehot = tf.one_hot(test_labels,num_output).eval()[0]
        #index is given by 
        imIndex = np.where(saved_label_onehot == 1)[0][0]
        # get the image from the array
        im = Image.fromarray(np.uint8((saved_image)*255))
        im.save("C:\\Users\\Sebastian\\Desktop\\MLChinese\\tmp\\{}_{}.jpeg".format(imIndex,i))
        
        
