#!/usr/bin/env python
# coding: utf-8

activation_function1 = 'relu'
activation_function2 = 'tanh'
batch_size = 8
epochs = 100
optimiser = 'adam'

import keras
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Lambda,Add
from keras.utils import to_categorical
import tensorflow as tf

from keras.layers import Reshape

from keras import backend as K, Sequential
from keras import regularizers, optimizers
#get_ipython().magic(u'matplotlib inline')

from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint


import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import re
from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub

import os
cwd = os.getcwd()

#Function to convert string to int
def atoi(text) :
    return int(text) if text.isdigit() else text


# Function used to specify the key for sorting filenames in the directory
# Split the input based on presence of digits
def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

# Sorting the files in the directory specified based on filenames
root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("DenoisedTrain"):
    filenames.sort(key = natural_keys)
    rootpath = root
# print('Filenames: ',filenames)

# Reads the images as per the sorted filenames and stores it in a list
images = []
for filename in filenames :
    filepath = os.path.join(root,filename)
    image = ndimage.imread(filepath, mode = "L")
    images.append(image)
    print(filename)

print('Total Images: ',len(images))

#Loading the labels for resized cropped images
labels = np.load('resized_cropped_labeledimages.npy')
labels_list = []
for i in range(len(labels)):
    labels_list.append(labels[i])
print(labels.shape)


# In[11]:


train_labels = np.zeros((770,216,64,8))


#Loop to perform one-hot encoding for the labels
for i in range(len(labels_list)) :
    for j in range(216) :
        for k in range(64):
            # train_labels[i][j][k][labels_list[i][j][k]] = 1
            if(labels_list[i][j][k] == 0):
                train_labels[i][j][k][0] = 1
            elif(labels_list[i][j][k] == 1):
                train_labels[i][j][k][1] = 1
            elif(labels_list[i][j][k] == 2):
                train_labels[i][j][k][2] = 1
            elif(labels_list[i][j][k] == 3):
                train_labels[i][j][k][3] = 1
            elif(labels_list[i][j][k] == 4):
                train_labels[i][j][k][4] = 1
            elif(labels_list[i][j][k] == 5):
                train_labels[i][j][k][5] = 1
            elif(labels_list[i][j][k] == 6):
                train_labels[i][j][k][6] = 1
            elif(labels_list[i][j][k] == 7):
                train_labels[i][j][k][7] = 1

images=np.array(images)
print(images.shape)
images = images.reshape(images.shape[0],216,64,1)

print(images.shape)
#Generate a random train set from 770 indices
train_indices = np.random.choice(770,500,replace = False)
print(sorted(train_indices))
train_images_random = []
train_labels_random = []

#Create the train set (images and labels) based on the randomly generated train indices
for i in train_indices:
    train_images_random.append(images[i])
    train_labels_random.append(train_labels[i])

#Generate the test set from the original image list by excluding the train indices

test_indices = [x for x in range(770) if x not in train_indices]
print(test_indices)
test_images = []
test_labels = []
for i in test_indices:
    test_images.append(images[i])
    test_labels.append(train_labels[i])


train_images = np.array(train_images_random)
train_labels = np.array(train_labels_random)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Convert the train and test lists (train images, train labels, test images, test labels) to numpy arrays of type float32
train_images = train_images.astype('float32')
train_labels = train_labels.astype('float32')
test_images = test_images.astype('float32')
test_labels = test_labels.astype('float32')


#Set the input image shape to 216x500
data_shape = 216*64
#Set the weight decay parameter for Frobenius norm to 0.001
weight_decay = 0.0001

""" Model definition for retinal layer segmentation using dilated convolutions"""

"""
Each encoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and a Max pooling layer
Each dilation layer consists of a dilated convolutional filter with Batch Normalization layer and ReLU activation
Each decoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and Upsampling layer 
Additive skip connections transfer the features from encoder to the corresponding decoder blocks respectively
Classification path consists of a convolutional layer of kernel size 1x1 and a Softmax activation function
"""
inputs = Input(shape=(216,64,1))
# print(inputs)
#Group 1

# model = Sequential()
# model.add()

#First block
print('Group 1')
print('First block')
print('Fhalf')

print('inputsize: ',inputs.shape)
L11fh1 = Conv2D(64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(inputs)
print('L11fh1 shape: ',L11fh1.shape)
L11fh2 = Conv2D(64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L11fh1)
#L2 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L1)
print('L11fh2 shape: ',L11fh2.shape)
L11fh3 = BatchNormalization()(L11fh2)
print('L11fh3 shape: ',L11fh3.shape)
L11fh4 = Activation(activation_function2)(L11fh3)
print('L11fh4 shape: ',L11fh4.shape)
L11fh5 = MaxPooling2D(pool_size=(2,2))(L11fh4)
print('L11fh5 shape: ',L11fh5.shape)

print('Shalf')

#resize using Convultion is used as our image have no homogeneous type data.

L11sh1 = Conv2D(64,kernel_size=(107,31),kernel_regularizer=regularizers.l2(weight_decay))(inputs)
print('L11sh1(resized) shape: ',L11sh1.shape)

L11sh2 = Conv2D(64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L11sh1)
print('L11sh2 shape: ',L11sh2.shape)

L11sh3 = Conv2D(64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L11sh2)
print('L11sh3 shape: ',L11sh3.shape)

#concatenate L11fh5 and L11sh3
print('Output')
L11o1 = keras.layers.concatenate([L11fh5, L11sh3], 3)
print('L11o1 shape: ',L11o1.shape)

#Second block
print('Second block')
print('Fhalf')
print('inputsize: ',L11o1.shape)
L12fh1 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L11o1)
print('L12fh1 shape: ',L12fh1.shape)
L12fh2 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L12fh1)
#L2 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L1)
print('L12fh2 shape: ',L12fh2.shape)
L12fh3 = BatchNormalization()(L12fh2)
print('L12fh3 shape: ',L12fh3.shape)
L12fh4 = Activation(activation_function2)(L12fh3)
print('L12fh4 shape: ',L12fh4.shape)
L12fh5 = MaxPooling2D(pool_size=(2,2))(L12fh4)
print('L12fh5 shape: ',L12fh5.shape)

print('Shalf')

#resize using Convultion is used as our image have no homogeneous type data.

L12sh1 = Conv2D(128,kernel_size=(162,48),kernel_regularizer=regularizers.l2(weight_decay))(inputs)
print('L12sh1(resized) shape: ',L12sh1.shape)

L12sh2 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L12sh1)
print('L12sh2 shape: ',L12sh2.shape)

L12sh3 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L12sh2)
print('L12sh3 shape: ',L12sh3.shape)

#concatenate L12fh5 and L12sh3
print('Output')
L12o1 = keras.layers.concatenate([L12fh5, L12sh3], 3)
print('L12o1 shape: ',L12o1.shape)
##Third block
#print('Third block')
#print('Fhalf')
#print('inputsize: ',L12o1.shape)
#L13fh1 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L12o1)
#print('L13fh1 shape: ',L12fh1.shape)
#L13fh2 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L13fh1)
##L2 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L1)
#print('L13fh2 shape: ',L13fh2.shape)
#L13fh3 = BatchNormalization()(L13fh2)
#print('L13fh3 shape: ',L13fh3.shape)
#L13fh4 = Activation(activation_function2)(L13fh3)
#print('L13fh4 shape: ',L13fh4.shape)
#L13fh5 = MaxPooling2D(pool_size=(2,2))(L13fh4)
#print('L13fh5 shape: ',L13fh5.shape)
#
#print('Shalf')
#
##resize using Convultion is used as our image have no homogeneous type data.
#
#L13sh1 = Conv2D(256,kernel_size=(190,57),kernel_regularizer=regularizers.l2(weight_decay))(inputs)
#print('L13sh1(resized) shape: ',L13sh1.shape)
#
#L13sh2 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L13sh1)
#print('L13sh2 shape: ',L13sh2.shape)
#
#L13sh3 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L13sh2)
#print('L13sh3 shape: ',L13sh3.shape)
#
##concatenate L12fh5 and L12sh3
#print('Output')
#L13o1 = keras.layers.concatenate([L13fh5, L13sh3], 3)
#print('L13o1 shape: ',L13o1.shape)

#Group 2
print('\nGroup 2')
print('inputsize: ',L12o1.shape)
L211 = Conv2D(512,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L12o1)
print('L211 shape: ',L211.shape)
L21o = Conv2D(512,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L211)
#L2 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L1)
print('L21o shape: ',L21o.shape)

#Group 3
print('\nGroup 3')
#First block
print('First block')
print('Fhalf')
# L31fh1 = UpSampling2D( size = (2,2)) (L21o)
L31fh1 = Deconvolution2D(256, kernel_size = (7,7), activation = activation_function2,name = "ct_deconv_l31fh1")(L21o)
# (3,3) +2,+2
print('L31fh1 shape: ',L31fh1.shape)
L31fh2 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L31fh1)
# (3,3) -2,-2
print('L31fh2 shape: ',L31fh2.shape)
L31fh3 = Activation(activation_function1)(L31fh2)
print('L31fh3 shape: ',L31fh3.shape)
L31fh4 = Conv2D(256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L31fh3)
print('L31fh4 shape: ',L31fh4.shape)
L31fh5 = Activation(activation_function1)(L31fh4)
print('L13fh5 shape: ',L31fh5.shape)
L31fh6 = Deconvolution2D(256, kernel_size = (7,7), activation = activation_function2,name = "ct_deconv_l31fh6")(L31fh5)
# L31fh6 = UpSampling2D( size = (2,2)) (L31fh5)
print('L31fh6 shape: ',L31fh6.shape)
L31fh7 = BatchNormalization()(L31fh6)
print('L31fh7 shape: ',L31fh7.shape)

print('second half')
L31sh1 = Deconvolution2D(256, kernel_size = (5,5), activation = activation_function2,name = "ct_deconv_l31sh1")(L12o1)
print('L11o1 shape:',L12o1.shape)
# L31sh1 = UpSampling2D( size = (2,2))(L12o1)
# print('L31sh1 shape: ',L31sh1.shape)
print('L31sh1 shape: ',L31sh1.shape) #(55,17,256)
L31sh2 = keras.layers.concatenate([L31sh1, L31fh7], 3)
print('L31ot shape: ',L31sh2.shape)
L31o = Conv2D(256, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2,padding="same")(L31sh2)
print('L31ot shape: ',L31o.shape) #(55,17,256)

#Second block
print('Second block')
print('Fhalf')
L32fh0 = UpSampling2D( size = (2,2)) (L31o)
print('L32fh0 shape: ',L32fh0.shape)
L32fh1 = Deconvolution2D(128, kernel_size = (3,3), activation = activation_function2,name = "ct_deconv_l32fh1")(L32fh0)
# (3,3) +2,+2
print('L32fh1 shape: ',L32fh1.shape)
L32fh2 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L32fh1)
# (3,3) -2,-2
print('L32fh2 shape: ',L32fh2.shape)
L32fh3 = Activation(activation_function1)(L32fh2)
print('L32fh3 shape: ',L32fh3.shape)
L32fh4 = Conv2D(128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay))(L32fh3)
print('L32fh4 shape: ',L32fh4.shape)
L32fh5 = Activation(activation_function1)(L32fh4)
print('L32fh5 shape: ',L32fh5.shape)
L32fh6 = Deconvolution2D(128, kernel_size = (3,3), activation = activation_function2,name = "ct_deconv_l32fh6")(L32fh5)
# L32fh6 = UpSampling2D( size = (2,2)) (L31fh5)
print('L32fh6 shape: ',L32fh6.shape)
L32fh7 = BatchNormalization()(L32fh6)
print('L32fh7 shape: ',L32fh7.shape)

print('second half')
L32sh1 = Deconvolution2D(128, kernel_size = (5,5), activation = activation_function2,name = "ct_deconv_l32sh1")(L11o1)
print('L11o1 shape:',L11o1.shape)
# L31sh1 = UpSampling2D( size = (2,2))(L12o1)
# print('L31sh1 shape: ',L31sh1.shape)
print('L32fh7 shape: ',L32fh7.shape)
L32sh2 = keras.layers.concatenate([L32sh1, L32fh7], 3)
print('L32sh2 shape: ',L32sh2.shape)
L32o = Conv2D(128, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2,padding="same")(L32sh2)
print('L32ot shape: ',L32o.shape) #(110,34,128)

#Group 4
print('\nGroup 4')
#First block
print('First block')
print('Fhalf')
L41fh1 = UpSampling2D( size = (2,2))(L32o)
print('L41fh1 shape: ',L41fh1.shape)
# L41fh1 = Deconvolution2D(64, kernel_size = (3,3), strides=(2,2), activation = activation_function2,name = "ct_deconv_41fh1",padding="same")(L32o)
# print('L41sh1 shape:',L41fh1.shape)
L41fo = Conv2D(64, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2)(L41fh1)
print('L41fo shape: ',L41fo.shape)

print('Shalf')
L41sh1 = Dropout(0.2)(L41fo)
print('L41sh1 shape: ',L41sh1.shape)
L41so = Conv2D(8, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2)(L41sh1)
print('L41fo(pa1) shape: ',L41so.shape)

#Second block
print('Second block')
print('Fhalf')
L42fh1 = UpSampling2D( size = (4,4))(L31o)
print('L41fh1 shape: ',L41fh1.shape)
# L42fh1 = Deconvolution2D(256, kernel_size = (3,3), strides=(4,4), activation = activation_function2,name = "ct_deconv_42fh1",padding="same")(L31o)
# print('L42sh1 shape:',L42fh1.shape)
L42fo = Conv2D(128, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2)(L42fh1)
print('L42fo shape: ',L42fo.shape)

print('Shalf')
L42sh1 = Dropout(0.2)(L42fo)
print('L42sh1 shape: ',L42sh1.shape)
L42so = Conv2D(8, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2)(L42sh1)
print('L42fo(pa2) shape: ',L42so.shape)

#Group 5
print('\nGroup 5')
#only one block
L51sh1 = keras.layers.concatenate([L41fo, L42fo], 3)
print('L51sh2 shape: ',L51sh1.shape)
L51sh2 = BatchNormalization()(L51sh1)
L51sh3 = Activation('relu')(L51sh2)
L51sh4 = Dropout(0.2)(L51sh3)
print('L51sh2 shape: ',L51sh4.shape)

#Classification block
L51o = Conv2D(8, kernel_size = (3,3), kernel_regularizer=regularizers.l2(weight_decay), activation = activation_function2)(L51sh4)
print('L51o(p0) shape: ',L51o.shape)

Lreshape = Reshape((data_shape,8),input_shape = (216,64,8))(L51o)
print('Lreshape shape: ',Lreshape.shape)
Lout = Activation('softmax')(Lreshape)

print('Lout(Final Output) shape: ',Lout.shape)
model = Model(inputs = inputs, outputs = Lout)
model.summary()

"""End of model defination"""

# Load the pre-trained weights if already trained
# Already trained weights are available in Model_weights/
# model.load_weights("RelaynetO_5.hdf5")

# from keras.utils import plot_model
# plot_model(model, to_file='model2_add_up.png',show_shapes= True)
# Load the weighted images obtained after pre-processing
weights = np.load('weighted_cropped_images.npy')

weights.shape

np.unique(weights)

weights_matrix = []
for i in train_indices:
    weights_matrix.append(weights[i])

sample_weights = np.array(weights_matrix)

sample_weights = np.reshape(sample_weights, (500, data_shape))

# Smoothing parameter for computation of dice co-efficient
train_labels = np.reshape(train_labels, (500, data_shape, 8))
test_labels = np.reshape(test_labels, (270, data_shape, 8))
smooth = 1

# Calculation of the dice co-efficient based on actual and predicted labels
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice loss computed as -dice co-efficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Combined loss of weighted multi-class logistic loss and dice loss
def customized_loss(y_true, y_pred):
    return (1 * K.categorical_crossentropy(y_true, y_pred)) + (0.5 * dice_coef_loss(y_true, y_pred))

# Using SGD optimiser with Nesterov momentum and a learning rate of 0.001
# optimiser = optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True)
# optimiser = 'Adam'

# Compiling the model
model.compile(optimizer=optimiser, loss=customized_loss, metrics=['accuracy', dice_coef], sample_weight_mode='temporal')

# Defining Callback functions which will be called by model during runtime when specified condition is satisfied
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger('exp1.csv')
model_checkpoint = ModelCheckpoint("exp1.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

"""Train the model by specifying all the required arguments
Sample weights are passed to the sample_weight argument - Numpy array of weights for the training samples, 
used for weighting the loss function (during training only)
"""

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels),
          sample_weight=sample_weights, callbacks=[lr_reducer, csv_logger, model_checkpoint])

def test_preprocessing(test_image):
    test_image = np.squeeze(test_image, axis=2)
    test_image = test_image.reshape((1, 216, 64, 1))
    return test_image

# Computation of the layer-wise dice scores
def dice_score(layer, y_true, y_pred):
    y_true_layer = y_true[:, layer]
    y_pred_layer = y_pred[:, layer]
    # print(y_true_layer.shape)
    # print(y_pred_layer.shape)
    intersection = np.dot(y_true_layer, y_pred_layer.T)
    score = 2. * intersection / (np.sum(y_true_layer) + np.sum(y_pred_layer))
    return score

test_images_size = len(test_images)
test_images_size

# Run the test set on the trained model and compute the layer wise dice scores for each test image
test_dice_scores = np.zeros((test_images_size, 8))
for image_no in range(test_images_size):
    prediction = model.predict(test_preprocessing(test_images[image_no]))
    # print(prediction.shape)
    prediction = np.squeeze(prediction, axis=0)
    # print(prediction.shape)
    print(image_no)
    for layer_no in range(8):
        test_dice_scores[image_no][layer_no] = dice_score(layer_no, test_labels[image_no], prediction)

# Compute the mean dice score over all images for each of the retinal layers
overall_dice_scores = np.zeros((8))
for layer_no in range(8):
    overall_dice_scores[layer_no] = np.mean(test_dice_scores[:, layer_no])

print(overall_dice_scores)

