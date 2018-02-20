# -*- coding: utf-8 -*-
"""
Input TFRecord
"""

import tensorflow as tf
import time
import math #to use radians in rotating the image
import random
from classDataManip import makeDir
import os

#%%
inputDim = 48

#%%

def decodeTrain(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={'train/image': tf.FixedLenFeature([], tf.string),
                  'train/label': tf.FixedLenFeature([], tf.int64)})
    # tfrecords is saved in raw bytes, need to convert this into usable format
    # May want to save this as tf.float32???
    image = tf.decode_raw(features['train/image'], tf.uint8)
    # Reshape image data into the original shape (try different forms)
#    image1 = tf.reshape(image, [inputDim, inputDim, 1]); #2D no batch
#    image2 = tf.reshape(image, [inputDim**2,1]);         #1D no batch
    """Try with no '1' on the end of array (which denotes RGB or greyscale)"""
    image1 = tf.reshape(image, [inputDim, inputDim]); #2D no batch
    image2 = tf.reshape(image, [inputDim**2]);         #1D no batch
    print(image1)
    print(image2)
    # Cast label data
    label = tf.cast(features['train/label'], tf.int32)
    return image2, label

def decodeTest(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={'test/image': tf.FixedLenFeature([], tf.string),
                  'test/label': tf.FixedLenFeature([], tf.int64)})
    # tfrecords is saved in raw bytes, need to convert this into usable format
    # May want to save this as tf.float32???
    image = tf.decode_raw(features['test/image'], tf.uint8)
    # Reshape image data into the original shape (try different forms)
#    image1 = tf.reshape(image, [inputDim, inputDim, 1]); #2D no batch
#    image2 = tf.reshape(image, [inputDim**2,1]);         #1D no batch
    """Try with no '1' on the end of array (which denotes RGB or greyscale)"""
    image1 = tf.reshape(image, [inputDim, inputDim]); #2D no batch
    image2 = tf.reshape(image, [inputDim**2]);         #1D no batch
    print(image1)
    print(image2)
    # Cast label data
    label = tf.cast(features['test/label'], tf.int32)
    return image2, label

def augment(image, label):
    """Apply distortions to the image, here rotation and translation
    Not included yet"""
    #rotate image
    degree_angle = random.randint(-10,10) # random integer from -10 to 10
    radian = degree_angle * math.pi / 180 #convert to radians
    #reshape the image so it has 2D shape
    image = tf.contrib.image.rotate(tf.reshape(image, [inputDim, inputDim]),radian)
    image = tf.reshape(image, [inputDim**2])
    #translate image
    #translate_x = random.randint(-3,3) #random integer between -3 and 3
    #translate_y = random.randint(-3,3) #this denotes translation in x and y
    #tf.contrib.image.translate(image,(translate_x,translate_y))
    #also can scale images using
    #tf.image.resize_images
    #need to look at the documentation for these methods, see if we have
    # the correct arguments
    return image, label

def normalize(image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label

# Creates a dataset that reads all of the examples from two files.
def inputs(trainType,tfrecord_filename,batch_size,num_epochs,normalize=True):
    """If num_epochs is set to 0, repeat infinitely"""
    #filenames = [tfrecord_filename]
    filenames = tfrecord_filename
    dataset = tf.data.TFRecordDataset(filenames)
    if num_epochs == 0: #if set to 0, repeat infinitely
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)
    print(dataset)
    
    if trainType == 'train':
        dataset = dataset.map(decodeTrain)  # Parse the record into tensors.
        #dataset = dataset.map(augment)
    elif trainType == 'test':
        #do not augment testing data! Only need to augment training data
        dataset = dataset.map(decodeTest)
    else:
        raise ValueError("trainType not specified properly as train or test")
    
    if normalize == True:
        dataset=dataset.map(normalize) #normalize the image values to be between -0.5 and 0.5
    dataset = dataset.shuffle(1000 + 3 * batch_size) #shuffle the order of the images
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()