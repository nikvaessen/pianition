#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np_utils
import os
from os.path import isfile
from PIL import Image as Img
from datetime import datetime

import keras
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, MaxPooling2D, Reshape
from keras.layers import Conv2D, BatchNormalization, Lambda, Permute, GRU
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

import sys
sys.path.insert(0,'..')

from pianition.data_util import load_dataset
from pianition.keras_models import conv1d_gru, RNN


# In[2]:


def get_callbacks(checkpoint_name):
    logDir = "./Graph/" + checkpoint_name +datetime.now().strftime("%H%M%S")+ "/"
    tb = TensorBoard(log_dir=logDir,
                     histogram_freq=2,
                     write_graph=True,
                     write_images=True,
                     write_grads=True,
                     update_freq='epoch')

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

    callback_list = [checkpoint_callback, reducelr_callback, tb]

    return callback_list


# In[3]:


def trainer(network_name, dataset_path, epochs, batch_size, lr):
    # Data iterator
    dataset = load_dataset(dataset_path)
    train_x, train_y = dataset.get_train_full()
    val_x, val_y = dataset.get_val_full()
    input_shape = (train_x.shape[1], train_x.shape[2])
    #Create your model
    network = eval(network_name)
    print("Running ",network_name," with ", input_shape)
    train_model = network((train_x.shape[1], train_x.shape[2]))
    #compile
    train_model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr),
                        metrics=['accuracy'])
    #Train the model
    # Better to change checkpoint name before run
    train_model.fit(train_x,
                    train_y,
                    batch_size,
                    epochs,
                    validation_data=(val_x, val_y),
                    callbacks=get_callbacks(checkpoint_name=network_name+"_"+str(input_shape)))


# In[4]:


def run_exps(path = "../data/",dataset_type="full", epochs=1, batch_size=32, lr=0.001):
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            if(dataset_type in dir):
                current_data_path = os.path.join(path, dir)
                print(current_data_path)
                _ = trainer("conv1d_gru", current_data_path, epochs, batch_size, lr)
#                 tf.reset_default_graph()
                keras.backend.clear_session()
                _ = trainer("RNN", current_data_path, epochs, batch_size, lr)
                keras.backend.clear_session()


# In[5]:


_ = run_exps(path="../data/", dataset_type='debug')


# In[ ]:




