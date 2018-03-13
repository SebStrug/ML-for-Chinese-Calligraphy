# -*- coding: utf-8 -*-
"""
Generative Adversarial network with the discriminator only
"""

import os
import glob
import time
import itertools
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
from InputTFRecord import inputs
from classDataManip import makeDir

#reset the existing graph << start of any tensorflow code!
tf.reset_default_graph() 

# initialise the parameters
inputDim = 48
num_output = 10
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
images_out = 3  #images_out**2: number of test images produced
paramStruct = namedtuple("ModelParameters","inputDim num_output, \
                         batch_size, num_epochs, learning_rate, images_out")
parameters = paramStruct(inputDim, num_output, \
                         batch_size, num_epochs, learning_rate, images_out)
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data": True}
print(parameters); print("\n");

localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0' #local data path
savePath = 'C:/Users/Sebastian/Desktop/MLChinese/Saved_runs/' #path to save tensorboard data to
name_of_run = 'full_GAN' 

def generate_gif(): #run this after you're finished to generate a gif for the images produced
    images = []
    for filename in glob.glob(LOGDIR+'/Image_outputs' + '/*.png'): #assuming gif
        im=Image.open(filename)
        images.append(np.array(im))
    imageio.mimsave(LOGDIR + '/Image_outputs/' + 'generation_animation.gif', images, fps=3)

# Discriminator, D(x)
# reuse must be set to true by default
def discriminator(x, isTrain=True, reuse=False): #reuse=True by default?
    with tf.variable_scope('discriminator', reuse=reuse): #tf.AUTO_REUSE instead of reuse?
        # 1st hidden layer. Goes from 48x48 -> 24x24
        conv1 = tf.layers.conv2d(x, 12, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = tf.nn.leaky_relu(conv1)
        
        # 2nd hidden layer. Goes from 24x24 -> 12x12
        conv2 = tf.layers.conv2d(lrelu1, 24, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

        # 3rd hidden layer. Goes from 12x12 -> 6x6
        conv3 = tf.layers.conv2d(lrelu2, 48, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

        # 4th hidden layer. Goes from 6x6 -> 3x3
        conv4 = tf.layers.conv2d(lrelu3, 96, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain))

        # output layer. Goes from 3x3 -> 1x1
        # need VALID padding so it reduces to a 1x1
        conv5 = tf.layers.conv2d(lrelu4, 1, [3, 3], strides=(1, 1), padding='valid')

        return tf.nn.sigmoid(conv5), conv5

def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # lrelu is a 'leaky relu layer'
        # 1st hidden layer, uses TRANSPOSE i.e. deconvolutions
        conv1 = tf.layers.conv2d_transpose(x, 96, [3, 3], strides=(1, 1), padding='valid')
        lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 48, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 24, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 12, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain))

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')

        return tf.nn.tanh(conv5) 
    
def load_data(parameters, train_kwargs, localPath): #function to load in the data
    train_tfrecord_filename = localPath+'\\train'+str(parameters.num_output)+'.tfrecords'
    #test_tfrecord_filename = localPath+'\\test'+str(num_output)+'.tfrecords'
    #tfrecord_files = [train_tfrecord_filename, test_tfrecord_filename]
    tfrecord_files = [train_tfrecord_filename]
    image_batch, label_batch = inputs('train', tfrecord_files, parameters.batch_size, \
                                      parameters.num_epochs, **train_kwargs)
    """Can't incorporate the test files too right now because the decoder
    has a separate 'train/...' and 'test/...' option"""
    
    # calculate the total number of files
    numFiles = sum([sum(1 for _ in tf.python_io.tf_record_iterator(i)) for i in tfrecord_files])
    print("Handling {} total files for {} unique outputs, approximately {} samples per unique output\n".\
          format(numFiles, num_output, int(numFiles/num_output)))
    return image_batch, label_batch, numFiles   
        
def show_result(num_out, show = False, save = False, path = 'result.png'):
    fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    size_figure_grid = num_out #output 25 images in a 5x5 grid
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(num_out, num_out))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (48, 48)), cmap='gray')
    #label = 'Epoch {0}'.format(num_epoch)
    #fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()   
        
        
image_batch, label_batch, numFiles = load_data(parameters, train_kwargs, localPath)

#Define the placeholders for the images and labels
print("Defining placeholders...")
x = tf.placeholder(tf.float32, [None,inputDim**2], name="images")
#y_ = tf.placeholder(tf.float32, [None,num_output], name="labels") # holds the labels
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100), name="random") # holds random numbers for the generator
isTrain = tf.placeholder(dtype=tf.bool, name="train_or_not") # boolean for training or not

with tf.name_scope('reshape'):
    # reshape x to a 4D tensor with second and third dimensions being width/height
    # [None, inputDim, inputDim, 1] didn't work
    x_image = tf.reshape(x, [-1,inputDim,inputDim,1])

image_summary = tf.summary.image('input', x_image, 4) # Show 4 examples of output images on tensorboard

# networks : generator
G_z = generator(z, isTrain)
# networks: discriminator
D_real, D_real_logits = discriminator(x_image, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
with tf.name_scope("Losses"):
    with tf.name_scope("Discriminator_loss_real"):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                            logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        D_loss_real_summ = tf.summary.scalar("Discriminator_loss_real",D_loss_real)
    with tf.name_scope("Discriminator_loss_fake"):
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                            logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
        D_loss_fake_summ = tf.summary.scalar("Discriminator_loss_fake",D_loss_fake)
    with tf.name_scope("Discriminator_loss"):
        D_loss = D_loss_real + D_loss_fake
        D_loss_total_summ = tf.summary.scalar("Discriminator_loss_total",D_loss)
    with tf.name_scope("Generator_loss"):
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                            logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        G_loss_summ = tf.summary.scalar("Generator_loss",G_loss)
        
# have separate summaries for the discriminator and generator
D_summary = tf.summary.merge([D_loss_real_summ, D_loss_fake_summ, D_loss_total_summ])
G_summary = tf.summary.merge([G_loss_summ,image_summary])

# trainable variables for each network
print("Creating the variables...")
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
print("Creating the optimizer...")
with tf.name_scope("Train"):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(D_loss, var_list=D_vars)
        G_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(G_loss, var_list=G_vars)
    
# Create a saver to save these summary operations
saver = tf.train.Saver()
#this is the operation that initialises all the graph variables
init_op = tf.group(tf.global_variables_initializer(),\
                   tf.local_variables_initializer()) 

# open session and initialize all variables
print("Running the training...")
np.random.seed(int(time.time()))

LOGDIR = makeDir(savePath,name_of_run,num_output,learning_rate,batch_size)
os.makedirs(LOGDIR+'/Image_outputs/')

with tf.Session() as sess: # or tf.InteractiveSession() ?
    # Create writers
    print("Creating the writers...")
    gen_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/generator')
    gen_writer.add_graph(sess.graph)
    discrim_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/discriminator')
    discrim_writer.add_graph(sess.graph)
    
    print("Initialising the variables...")
    sess.run(init_op)
    start_time = time.time()
    step = 0
    try:
        while True:
            # update discriminator
            x_, train_labels = sess.run([image_batch, label_batch])
            #print("Output images: {}".format(x_))
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            loss_d_, _, discrim_summary = sess.run([D_loss, D_optim, D_summary], \
                                                   {x: x_, z: z_, isTrain: True})
            print("Step: {}, Discriminator loss: {:.3}".format(step, loss_d_))
            
            # update generator
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            loss_g_, _, gen_summary = sess.run([G_loss, G_optim, G_summary], \
                                               {z: z_, x: x_, isTrain: True})
            print("Step: {}, Generator loss: {:.3}".format(step, loss_g_))
            
            if step % 10 == 0: ## epoch = math.floor(numFiles/batch_size)
                print("Showing a generated result for step {}...".format(step))
                print("Currently at epoch {:.1}".format(numFiles/(parameters.batch_size*step)))
                fixed_p = LOGDIR + '/Image_outputs/' + 'DCGAN_' + str(step) + '.png'
                show_result(images_out, show=True, save=True, path=fixed_p)
                
                discrim_writer.add_summary(discrim_summary, step)
                gen_writer.add_summary(gen_summary, step)
                
                
            if step % 20 == 0:
                saver.save(sess, os.path.join(LOGDIR, "DLoss{:.3}_GLoss{:.3}.ckpt".\
                                                  format(loss_d_, loss_g_)))        
            step += 1
            
    except tf.errors.OutOfRangeError:
            duration = time.time() - start_time
            print('Done {} epochs, {} steps, took {:.3} mins.'.\
                  format(num_epochs,step,duration/60))
    gen_writer.close()
    discrim_writer.close()    
    sess.close() 