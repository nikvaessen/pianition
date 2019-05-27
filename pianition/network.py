#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import os
from os.path import isfile
from PIL import Image as Img
from data_util import *
from datetime import datetime

import keras
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers

import librosa
import librosa.display
import matplotlib.pyplot as plt

import tensorflow as tf


# In[2]:


SHUFFLE_BUFFER = 1000
BATCH_SIZE = 32
NUM_CLASSES = 12

# Create a description of the features.  
feature_description = {
    'feature0': tf.FixedLenFeature([32768], tf.float32),
    'feature1': tf.FixedLenFeature([1], tf.int64)
}

def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.parse_single_example(example_proto, feature_description)
    parsed_example["feature0"] = tf.transpose(tf.reshape(parsed_example['feature0'], (256,128)))
    return parsed_example

def create_dataset(filepath):
    
    dataset = tf.data.TFRecordDataset(filepath)
    
    dataset = dataset.map(_parse_function) #, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.batch(BATCH_SIZE)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    feature = iterator.get_next()
    #print(feature)
    lmfcc = feature["feature0"]
    label = feature["feature1"]
    
    # Bring your picture back in shape
    lmfcc = tf.reshape(lmfcc, [-1,128, 256])
    
    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)
    print(lmfcc.shape)
    print(label.shape)

    return lmfcc, label


# In[3]:


lmfcc, label = create_dataset("../data/debug/sample.tfrecords")


# In[18]:


def get_callbacks(checkpoint_name):
    logDir = "./Graph/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    tb = TensorBoard(log_dir=logDir,
                     histogram_freq=2,
                     write_graph=True,
                     write_images=True,
                     write_grads=True,
                     update_freq='epoch')

#     tb_callback = TensorBoard(
#         log_dir='../models/logs/',
#         histogram_freq=1,
#         batch_size=32,
#         write_graph=True,
#         write_grads=False,
#         write_images=False,
#         embeddings_freq=0,
#         embeddings_layer_names=None,
#         embeddings_metadata=None,
#     )

    checkpoint_callback = ModelCheckpoint('../models/' + checkpoint_name +
                                          '{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max')

    reducelr_callback = ReduceLROnPlateau(monitor='val_acc',
                                          factor=0.5,
                                          patience=10,
                                          min_delta=0.01,
                                          verbose=1)

    callback_list = [checkpoint_callback, reducelr_callback]

    return callback_list


# In[19]:


# Data iterator
lmfcc, label = create_dataset("../data/debug/train.tfrecords")
lmfcc_val, label_val = create_dataset("../data/debug/val.tfrecords")

#Build network
NUM_CLASSES = 12  # Must Change in the tf reader as well
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 56
LSTM_COUNT = 96
NUM_HIDDEN = 64
L2_regularization = 0.001

# Input
model_input = keras.layers.Input(tensor=lmfcc)

for i in range(N_LAYERS):
    # give name to the layers

    layer = Conv1D(
        filters=CONV_FILTER_COUNT,
        kernel_size=FILTER_LENGTH,
        kernel_regularizer=regularizers.l2(L2_regularization),  # Tried 0.001
        name='convolution_' + str(i + 1))(model_input)

    layer = BatchNormalization(momentum=0.9)(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.4)(layer)

    ## LSTM Layer
    layer = LSTM(LSTM_COUNT, return_sequences=False)(layer)
    layer = Dropout(0.4)(layer)

    ## Dense Layer
    layer = Dense(NUM_HIDDEN,
                  kernel_regularizer=regularizers.l2(L2_regularization),
                  name='dense1')(layer)
    layer = Dropout(0.4)(layer)

    ## Softmax Output
    layer = Dense(NUM_CLASSES)(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer

model_output = Dense(NUM_CLASSES, activation='relu')(model_output)

#Create your model
train_model = Model(inputs=model_input, outputs=model_output)

#compile
train_model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001),
                    metrics=['accuracy'],
                    target_tensors=[label])

#Train the model
#steps per epoch could be viewed as dataset/batchsize
batch_size = 16
# Better to change checkpoint name before run
train_model.fit(epochs=70,
                steps_per_epoch=100,
                validation_data=(lmfcc_val, label_val),
                validation_steps=100,
                callbacks=get_callbacks(checkpoint_name="trail"))


# In[ ]:




