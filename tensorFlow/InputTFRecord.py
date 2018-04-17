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
import numpy as np

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
    #image1 = tf.reshape(image, [inputDim, inputDim]); #2D no batch
    #image2 = tf.reshape(image, [inputDim**2]);         #1D no batch
    #print(image1)
    #print(image2)
    # Cast label data
    label = tf.cast(features['train/label'], tf.int32)
    return image, label

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
    #reshape the image so it has 2D shape
    image = tf.reshape(image, [inputDim, inputDim,1])
    
    #adjust brightness
    image = tf.image.random_brightness(image,0.1)
    
    #rotate image
    degree_angle = random.randint(-10,10) # random integer from -10 to 10
    print("Rotation by {} degrees".format(degree_angle))
    radian = degree_angle * math.pi / 180 #convert to radians
    image = tf.contrib.image.rotate(image,radian)
    
    #translate image
    translate_x = int(np.random.normal(0)) #random integer normally distributed
    translate_y = int(np.random.normal(0)) #random integer normally distributed
    if abs(translate_x) > 2 or abs(translate_y) > 2:
        translate_x = 0 #so we don't get a shift of 5 
        translate_y = 0
    print("Translation by {},{} in x,y".format(translate_x,translate_y))
    image = tf.contrib.image.translate(image,(translate_x,translate_y))
    
    #scale the image
    #scaling doesn't seem to work with how small the lines are
#    resize_scale = inputDim + 0
#    image = tf.image.resize_images(image,tf.constant([inputDim+0,inputDim+5]))
#    image = tf.image.resize_image_with_crop_or_pad(image,inputDim,inputDim)
   
    #reshape the image back into 1D
    image = tf.reshape(image, [inputDim**2])
    
    return image, label

def normalize(image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats (normalisation in example)
#    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # More advanced normalisation that uses the mean and standard deviation
    image = tf.reshape(image, [inputDim, inputDim,1])
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [inputDim**2])
    return image, label

# Creates a dataset that reads all of the examples from two files.
def inputs(trainType, tfrecord_filename, batch_size, num_epochs,\
           normalize_images=False, augment_images=False, shuffle_data=False,\
           multiple_files=False):
    """If num_epochs is set to 0, repeat infinitely"""
    filenames = tfrecord_filename #[tfrecord_filename]??
    dataset = tf.data.TFRecordDataset(filenames)
    if num_epochs == 0: #if set to 0, repeat infinitely
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)
    print(dataset)
    
    if trainType == 'train':
        dataset = dataset.map(decodeTrain)  # Parse the record into tensors.
        if augment_images == True:   
            print("Augmenting the images")
            dataset = dataset.map(augment)
    elif trainType == 'test':
        #do not augment testing data! Only need to augment training data
        dataset = dataset.map(decodeTest)
    else:
        raise ValueError("trainType not specified properly as train or test")
    if normalize_images == True:
        dataset=dataset.map(normalize) #normalize the image values to be between -0.5 and 0.5
    if shuffle_data == True:
        dataset = dataset.shuffle(1000 + 3 * batch_size) #shuffle the order of the images
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()