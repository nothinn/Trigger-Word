'''
####ONLY USE NEEDED RAM
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)sess = tf.Session(config=config)set_session(sess)  # set this TensorFlow session as the default session for Keras
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#####
'''

import numpy as np

from keras.utils import multi_gpu_model

import os
import model
import utils

from keras.optimizers import Adam

'''

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

Ty = 1375 # The number of time steps in the output of our model


samplerate = 33100


model = model.model(input_shape = (Tx, n_freq))

print(model.summary())
model = multi_gpu_model(model)
print(model.summary())



opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# Load preprocessed training examples
X = np.load("Keras-Trigger-Word/XY_train/X.npy")
input(X.shape)
Y = np.load("Keras-Trigger-Word/XY_train/Y.npy")
input(Y.shape)


# Load preprocessed dev set examples
X_dev = np.load("Keras-Trigger-Word/XY_dev/X_dev.npy")
input(X_dev.shape)
Y_dev = np.load("Keras-Trigger-Word/XY_dev/Y_dev.npy")
input(Y_dev.shape)

'''
# Parameters
params = {'dim': (5511,101), #Dims = (samples, frequency bins)
          'batch_size': 64,
          'n_classes': 1, #Only detect one type of word
          'n_channels': 1375, #Number of output bins
          'shuffle': True}




#Load backgrounds
backgrounds = []

for f in os.listdir("data/backgrounds/"):
    (data, length) = utils.load_audio("data/backgrounds/" + f)
    utils.plt_values(data, "Original")
    hamming = utils.hamming_window_signal(data)
    utils.plt_values(hamming, "windowed")


    fft_orig = utils.fft(data)
    fft_hamm = utils.fft(hamming)
    utils.plt_values(fft_orig, "fft_orig")
    utils.plt_values(fft_hamm, "fft_hamm")
    




    print("Background was {} seconds".format(length) )

    backgrounds.append((data,length))






training_set = []
with open("data/training_act.txt") as f:
    lines = f.readlines() #Maybe need to exclude last line
    for line in lines:
        training_set.append((line, True))


with open("data/training_neg.txt") as f:
    lines = f.readlines() #Maybe need to exclude last line
    for line in lines:
        training_set.append((line, False))


validation_set = []
with open("data/training_act.txt") as f:
    lines = f.readlines() #Maybe need to exclude last line
    for line in lines:
        validation_set.append((line, True))


with open("data/training_neg.txt") as f:
    lines = f.readlines() #Maybe need to exclude last line
    for line in lines:
        validation_set.append((line, False))







print(backgrounds)

#Transformations
transformations = []
transformations.append(lambda input: input) #Do nothing


train_generator = model.DataGenerator(training_set, backgrounds, transformations)
valid_generator = model.DataGenerator(validation_set, backgrounds)

'''
for i in range(100):

    model.fit(X, Y, batch_size = 10, epochs=10)
    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)

    model.save('trains/trained_model_{}.h5'.format(i))
'''