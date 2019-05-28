# Authors: Sri Datta Budaraju, Peter 

import keras
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, MaxPooling2D, Reshape
from keras.layers import Conv2D, BatchNormalization, Lambda, Permute, GRU
from keras.layers import Bidirectional, concatenate
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers

def conv1d_gru(input_shape):
    #Build network
    NUM_CLASSES = 12  # Must Change in the tf reader as well
    N_LAYERS = 3
    CONV_FILTER_COUNT = 64
    FILTER_LENGTH = 5
    POOL_SIZE = 2
    GRU_COUNT = 64
    NUM_HIDDEN = 128
    L2_regularization = 0.001

    # Input
    model_input = keras.layers.Input(shape=input_shape)
    print("before permute ", model_input.shape)
    layer = Permute((2, 1), input_shape=(None, input_shape[0], input_shape[1]))(model_input)
    print("after permute ", layer.shape)

    # Conv1D , input_shape=(10, 128) for time series sequences of 10 time steps with 128 features per step
    # 1st conv
    layer = Conv1D(filters=CONV_FILTER_COUNT,
                   kernel_size=FILTER_LENGTH)(layer)  #(model_input)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_SIZE)(layer)
    layer = Dropout(0.2)(layer)

    for i in range(N_LAYERS - 1):
        layer = Conv1D(filters=128, kernel_size=FILTER_LENGTH)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_SIZE)(layer)
        layer = Dropout(0.4)(layer)

    ## LSTM Layer
    layer = GRU(GRU_COUNT, return_sequences=True)(layer)
    layer = GRU(GRU_COUNT, return_sequences=False)(layer)

    layer = Dropout(0.4)(layer)

    ## Softmax Output
    layer = Dense(NUM_CLASSES)(layer)
    layer = Activation('softmax')(layer)
    model_output = layer
    train_model = Model(inputs=model_input, outputs=(model_output))
    return train_model

def RNN(input_shape):

    NUM_CLASSES = 12  # Must Change in the tf reader as well
    N_LAYERS = 3
    CONV_FILTER_COUNT = 64
    FILTER_LENGTH = 5

    POOL_SIZE = 2

    GRU_COUNT = 64
    NUM_HIDDEN = 128
    L2_regularization = 0.001

    # Input
    model_input = keras.layers.Input(shape=input_shape)
    print(model_input.shape)
    layer = Permute((2, 1), input_shape=(input_shape[0], input_shape[1]))(model_input)
    print(layer.shape)

    ## LSTM Layer
    layer = LSTM(GRU_COUNT, return_sequences=True)(layer)
    layer = LSTM(GRU_COUNT, return_sequences=False)(layer)

    ## Softmax Output
    layer = Dense(NUM_CLASSES)(layer)
    layer = Activation('softmax')(layer)
    model_output = layer

    #Create your model
    train_model = Model(inputs=model_input, outputs=model_output)
    return train_model

def parallel(input_shape):

    N_LAYERS = 3
    FILTER_LENGTH = 5
    CONV_FILTER_COUNT = 56
    BATCH_SIZE = 32
    EPOCH_COUNT = 70
    NUM_HIDDEN = 64
    NUM_CLASSES = 12
    nb_filters1 = 16
    nb_filters2 = 32
    nb_filters3 = 64
    nb_filters4 = 64
    nb_filters5 = 64
    ksize = (3, 1)
    pool_size_1 = (2, 2)
    pool_size_2 = (4, 4)
    pool_size_3 = (4, 2)
    dropout_prob = 0.20
    dense_size1 = 128
    lstm_count = 64
    num_units = 120
    BATCH_SIZE = 64
    EPOCH_COUNT = 50
    L2_regularization = 0.001
    
    model_input = keras.layers.Input(shape=(input_shape))
    print(model_input.shape)
    reshaped_input = Reshape(target_shape=(input_shape[0], input_shape[1], 1))(model_input)
    ### Convolutional blocks
    conv_1 = Conv2D(filters=nb_filters1, kernel_size=ksize, strides=1,
                    padding='same', activation='relu', name='conv_1')(reshaped_input)
    pool_1 = MaxPooling2D(pool_size_1)(conv_1)

    conv_2 = Conv2D(filters=nb_filters2, kernel_size=ksize, strides=1,
                    padding='same', activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPooling2D(pool_size_1)(conv_2)

    conv_3 = Conv2D(filters=nb_filters3, kernel_size=ksize, strides=1,
                    padding='same', activation='relu', name='conv_3')(pool_2)
    pool_3 = MaxPooling2D(pool_size_1)(conv_3)

    conv_4 = Conv2D(filters=nb_filters4, kernel_size=ksize, strides=1,
                    padding='same', activation='relu', name='conv_4')(pool_3)
    pool_4 = MaxPooling2D(pool_size_2)(conv_4)

    conv_5 = Conv2D(filters=nb_filters5, kernel_size=ksize, strides=1,
                    padding='same', activation='relu', name='conv_5')(pool_4)
    pool_5 = MaxPooling2D(pool_size_2)(conv_5)

    flatten1 = Flatten()(pool_5)
    ### Recurrent Block
    # Pooling layer
    pool_lstm1 = MaxPooling2D(pool_size_3, name='pool_lstm')(reshaped_input)
    # Embedding layer
    squeezed = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(pool_lstm1)
    # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  # default merge mode is concat
    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name='concat')
    ## Softmax Output
    output = Dense(NUM_CLASSES, activation='softmax', name='preds')(concat)
    model_output = output
    model = Model(model_input, model_output)

    return model