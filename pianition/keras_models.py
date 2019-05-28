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
    layer = Permute((2, 1), input_shape=(None, 128, 256))(model_input)
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
    layer = Permute((2, 1), input_shape=(128, 256))(model_input)
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
