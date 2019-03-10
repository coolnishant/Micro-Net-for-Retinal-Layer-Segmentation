#!/usr/bin/env python
# coding: utf-8

# In[1]:

# [0.99011564 0.92536699 0.93408108 0.91316754 0.85130454 0.97386465 0.94672452 0.90012799]

import keras
from keras.layers import Activation
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

from keras import backend as K
from keras import regularizers, optimizers
#get_ipython().magic(u'matplotlib inline')


# In[2]:


from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint


# In[3]:


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


# In[4]:


import os
cwd = os.getcwd()


# In[5]:


#Function to convert string to int
def atoi(text) : 
    return int(text) if text.isdigit() else text


# In[6]:


# Function used to specify the key for sorting filenames in the directory
# Split the input based on presence of digits
def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]


# In[7]:


# Sorting the files in the directory specified based on filenames
root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("DenoisedTrain"):
    filenames.sort(key = natural_keys)
    rootpath = root


# In[8]:


# Reads the images as per the sorted filenames and stores it in a list
images = []
for filename in filenames :
    filepath = os.path.join(root,filename)
    image = ndimage.imread(filepath, mode = "L")
    images.append(image)
    print(filename)


# In[9]:


len(images)


# In[10]:


#Loading the labels for resized cropped images
labels = np.load('resized_cropped_labeledimages.npy')
labels_list = []
for i in range(len(labels)):
    labels_list.append(labels[i])
print(labels.shape)


# In[11]:


train_labels = np.zeros((770,216,64,8))


# In[12]:


#Loop to perform one-hot encoding for the labels
for i in range(len(labels_list)) :
    for j in range(216) :
        for k in range(64):
            if(labels_list[i][j][k] == 0):
                train_labels[i][j][k][0] = 1
            if(labels_list[i][j][k] == 1):
                train_labels[i][j][k][1] = 1
            if(labels_list[i][j][k] == 2):
                train_labels[i][j][k][2] = 1
            if(labels_list[i][j][k] == 3):
                train_labels[i][j][k][3] = 1
            if(labels_list[i][j][k] == 4):
                train_labels[i][j][k][4] = 1
            if(labels_list[i][j][k] == 5):
                train_labels[i][j][k][5] = 1
            if(labels_list[i][j][k] == 6):
                train_labels[i][j][k][6] = 1
            if(labels_list[i][j][k] == 7):
                train_labels[i][j][k][7] = 1


# In[13]:


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


# In[14]:


train_images = np.array(train_images_random)
train_labels = np.array(train_labels_random)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


# In[15]:


#Convert the train and test lists (train images, train labels, test images, test labels) to numpy arrays of type float32
train_images = train_images.astype('float32')
train_labels = train_labels.astype('float32')
test_images = test_images.astype('float32')
test_labels = test_labels.astype('float32')


# In[16]:


#Set the input image shape to 216x500
data_shape = 216*64
#Set the weight decay parameter for Frobenius norm to 0.001
weight_decay = 0.0001


# In[17]:


""" Model definition for retinal layer segmentation using dilated convolutions"""

"""
Each encoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and a Max pooling layer
Each dilation layer consists of a dilated convolutional filter with Batch Normalization layer and ReLU activation
Each decoder block consists of Convolutional layer, Batch Normalization layer, ReLU activation and Upsampling layer 
Additive skip connections transfer the features from encoder to the corresponding decoder blocks respectively
Classification path consists of a convolutional layer of kernel size 1x1 and a Softmax activation function
"""
inputs = Input(shape=(216,64,1))

#First encoder block
L1 = Conv2D(64,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(inputs)
L2 = BatchNormalization()(L1)
L2 = Activation('relu')(L2)
L3 = MaxPooling2D(pool_size=(2,2))(L2)

#Second encoder block
L4 = Conv2D(128,kernel_size=(3,3),dilation_rate= (2,2),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L3)
L5 = BatchNormalization()(L4)
L5 = Activation('relu')(L5)
L6 = MaxPooling2D(pool_size=(2,2))(L5)

#Third encoder block
L7 = Conv2D(128,kernel_size=(3,3),dilation_rate= (2,2),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L6)
L8 = BatchNormalization()(L7)
L9 = Activation('relu')(L8)

#Replacement of the third max pooling layer with 3 dilated convolutions with dilation rates 2,4,8 respectively
L10 = Conv2D(128,(3,3),dilation_rate= (2,2), padding = "same", activation='relu', name = "conv_dil_1")(L9)
L11 = BatchNormalization()(L10)
L12 = Activation('relu')(L11)
L13 = Conv2D(128,(3,3),dilation_rate= (4,4), padding = "same", activation='relu', name = "conv_dil_2")(L12)
L14 = BatchNormalization()(L13)
L15 = Activation('relu')(L14)
L16 = Conv2D(128,(3,3),dilation_rate= (8,8), padding = "same", activation='relu', name = "conv_dil_3")(L15)
L17 = BatchNormalization()(L16)
L18 = Activation('relu')(L17)

#Third decoder block corresponding to third encoder block
L19 = Conv2D(128,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay),
             name="skip_conv_1")(L6)
L19 = BatchNormalization()(L19)
L19 = Activation('relu')(L19)

#Additive connection 1
L20 = Add()([L18,L19])
L21 = UpSampling2D( size = (2,2)) (L20)


#L21 = Deconvolution2D(128, kernel_size = (3,3), strides = (2,2), activation = "relu", 
 #                     name = "ct_deconv_1", padding = "same")(L20)
    
#Second decoder block corresponding to second encoder block 
L21 = Conv2D(128,(3,3), padding = "same", kernel_regularizer=regularizers.l2(weight_decay))(L21)
L22 = BatchNormalization()(L21)
L23 = Activation('relu')(L22)
L24 = Conv2D(128,kernel_size=(3,3),padding = "same",kernel_regularizer=regularizers.l2(weight_decay),
             name="skip_conv_2")(L3)
L24 = BatchNormalization()(L24)
L24 = Activation('relu')(L24)

#Additive connection 1
L24 = Add()([L23,L24])
L25 = UpSampling2D(size = (2,2))(L24)

#First decoder block corresponding to first encoder block
L25 = Conv2D(64, (3,3), padding = "same", kernel_regularizer=regularizers.l2(weight_decay))(L25)
#L25 = Deconvolution2D(64, kernel_size = (3,3), strides = (2,2), activation = "relu", 
 #                     name = "ct_deconv_2", padding = "same")(L24)
#L25 = 
L26 = BatchNormalization()(L25)
L27 = Activation('relu')(L26)

#Classification block
L28 = Conv2D(8,kernel_size=(1,1),padding = "same",kernel_regularizer=regularizers.l2(weight_decay))(L27)
L29 = Reshape((data_shape,8),input_shape = (216,64,8))(L28)
L30 = Activation('softmax')(L29)
model = Model(inputs = inputs, outputs = L30)
model.summary()

"""End of model definition"""


# In[18]:


#Load the pre-trained weights if already trained
#Already trained weights are available in Model_weights/
model.load_weights("RelaynetO_5.hdf5")


# In[19]:


#from keras.utils import plot_model
#plot_model(model, to_file='model2_add_up.png',show_shapes= True)


# In[20]:


#Load the weighted images obtained after pre-processing
weights = np.load('weighted_cropped_images.npy')


# In[21]:


weights.shape


# In[22]:


np.unique(weights)


# In[23]:


weights_matrix = []
for i in train_indices:
    weights_matrix.append(weights[i])


# In[24]:


sample_weights = np.array(weights_matrix)


# In[25]:


sample_weights = np.reshape(sample_weights,(500,data_shape))


# 

# In[26]:


#Smoothing parameter for computation of dice co-efficient
train_labels = np.reshape(train_labels,(500,data_shape,8))
test_labels = np.reshape(test_labels,(270,data_shape,8))
smooth = 1


# In[27]:


#Calculation of the dice co-efficient based on actual and predicted labels
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[28]:


#Dice loss computed as -dice co-efficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[29]:


#Combined loss of weighted multi-class logistic loss and dice loss
def customized_loss(y_true,y_pred):
    return (1*K.categorical_crossentropy(y_true, y_pred))+(0.5*dice_coef_loss(y_true, y_pred))


# In[30]:


#Using SGD optimiser with Nesterov momentum and a learning rate of 0.001
optimiser = optimizers.SGD(lr = 0.005, momentum = 0.9, nesterov = True)
#optimiser = 'Adam'


# In[31]:


#Compiling the model 
model.compile(optimizer=optimiser,loss=customized_loss,metrics=['accuracy',dice_coef],sample_weight_mode='temporal')


# In[32]:


#Defining Callback functions which will be called by model during runtime when specified condition is satisfied
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger('exp1.csv')
model_checkpoint = ModelCheckpoint("exp1.hdf5",monitor = 'val_loss',verbose = 1,save_best_only=True)


# In[33]:


"""Train the model by specifying all the required arguments
Sample weights are passed to the sample_weight argument - Numpy array of weights for the training samples, 
used for weighting the loss function (during training only)
"""
model.fit(train_images,train_labels,batch_size=32,epochs=100,validation_data=(test_images,test_labels),sample_weight=sample_weights,callbacks=[lr_reducer, csv_logger,model_checkpoint])


# In[34]:


def test_preprocessing(test_image):
    test_image = np.squeeze(test_image,axis = 2)
    test_image = test_image.reshape((1,216,64,1))
    return test_image


# In[35]:


#Computation of the layer-wise dice scores
def dice_score(layer, y_true, y_pred) :
    y_true_layer = y_true[:,layer]
    y_pred_layer = y_pred[:,layer]
    #print(y_true_layer.shape)
    #print(y_pred_layer.shape)
    intersection = np.dot(y_true_layer,y_pred_layer.T)
    score = 2. * intersection/(np.sum(y_true_layer)+np.sum(y_pred_layer))
    return score
    


# In[36]:


test_images_size = len(test_images)
test_images_size


# In[37]:


#Run the test set on the trained model and compute the layer wise dice scores for each test image
test_dice_scores = np.zeros((test_images_size,8))
for image_no in range(test_images_size):
    prediction = model.predict(test_preprocessing(test_images[image_no])) 
    #print(prediction.shape)
    prediction = np.squeeze(prediction,axis = 0)
    #print(prediction.shape)
    print(image_no)
    for layer_no in range(8): 
        test_dice_scores[image_no][layer_no] = dice_score(layer_no,test_labels[image_no],prediction)


# In[38]:



# In[39]:


#Compute the mean dice score over all images for each of the retinal layers
overall_dice_scores = np.zeros((8))
for layer_no in range(8):
    overall_dice_scores[layer_no] = np.mean(test_dice_scores[:,layer_no])


# In[40]:


print(overall_dice_scores)


# In[ ]:




