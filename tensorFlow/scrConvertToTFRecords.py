# -*- coding: utf-8 -*-
"""
From https://github.com/blondegeek
"""

import tensorflow as tf
import numpy as np

# This was useful
# see http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow

# Example numpy file.
files = ['CH_7381_8380_1478761500.9.npz']

def _int64list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class NumpyToTFHelper(object):
    def __init__(self,filenames):
        self.filenames = filenames
        self.current_file = -1
        self.current_index = 0 
        self.current_array = None
    def get_next(self):
        if self.current_file == -1: 
            self.current_file += 1
            self.current_array = np.load(self.filenames[self.current_file])['arr_0']
        elif self.current_index == self.current_array.shape[0]-1:
            if self.current_file == len(self.filenames) - 1:
                return None
            self.current_file += 1
            self.current_array = np.load(self.filenames[self.current_file])['arr_0']
            self.current_index = 0 
        self.current_index += 1 
        # return single 16000 length record
        cur = self.current_array[self.current_index -1] 
        print(cur.shape)
        print(type(cur))
        return np.matrix(cur).A1

filename = "CH_int_array_TEST.tfrecords"
print('Writing', filename)
writer = tf.python_io.TFRecordWriter(filename)
N = NumpyToTFHelper(files) 
n = N.get_next()
while n is not None:
    raw = n.tostring()
    # remember to read back to numpy array, will need to specify that int was used.
    example = tf.train.Example(features=tf.train.Features(feature={
        'molecule':_int64list_feature(n)})) 
    writer.write(example.SerializeToString())
    n = N.get_next()
writer.close()