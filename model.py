from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Conv2D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten
from keras.backend import expand_dims

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

    print("Input shape ", input_shape)

    constraints = keras.constraints.MinMaxNorm(min_value = -1, max_value = 1, rate=1.0)

    print(X_input.shape)

    X = Reshape((1,input_shape[0], input_shape[1]), input_shape=input_shape)(X_input)
    #MODEL
    print(X_input.shape)

    X = Conv2D(16, kernel_size=(1,15), strides=4,kernel_constraint=constraints, bias_constraint=constraints)(X)       
    print(X.shape)                              # CONV1D
    #X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    print(X.shape)                              # CONV1D

    X = Dropout(0.2)(X)                                 # dropout (use 0.8)
    print(X.shape)                              # CONV1D

    X = Conv2D(4, kernel_size=(1,15), strides=4)(X)
    print(X.shape, "second conv")                              # CONV1D

    #print("Shape of X: ",X.shape)
    #X = GRU(units = num_classes, return_sequences = False)(X_input)   # GRU (use 128 units and return the sequences)
    #X = Dropout(0.2)(X)                                 # dropout (use 0.8)
    #X = BatchNormalization()(X)                                 # Batch normalization
    
    #print("Shape of X: ",X.shape)


    #X = GRU(units = num_classes)(X)

    X = Flatten()(X)
    print(X.shape)                              # CONV1D


    X = Dense(num_classes, activation="softmax")(X)
    print(X.shape)                              # CONV1D


    #print("Shape of X: ",X.shape)
    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)
    
    return model

