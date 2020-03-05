from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten


import numpy as np
import keras

import random

import utils




def model(input_shape, num_classes):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape = input_shape)

    #MODEL
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.2)(X)                                 # dropout (use 0.8)
    #print("Shape of X: ",X.shape)
    X = GRU(units = 128, return_sequences = False)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.2)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    #print("Shape of X: ",X.shape)


    #X = GRU(units = num_classes)(X)

    X = Dense(num_classes, activation="sigmoid")(X)

    #print("Shape of X: ",X.shape)
    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)
    
    return model

