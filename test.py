import numpy as np
import utils

import os

import datagenerator

import model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


####ONLY USE NEEDED RAM
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)sess = tf.Session(config=config)set_session(sess)  # set this TensorFlow session as the default session for Keras
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#####



fft = datagenerator.get_fft(np.sin(np.linspace(-np.pi*10,np.pi, 128)))

# Parameters
params = {'dim': (625,128), #Dims = (timesteps, frequency bins)
          'batch_size': 64,
          'n_classes': 1, #Only detect one type of word
          'Ty': 153, #Number of output bins
          'shuffle': True}



#list backgrounds
backgrounds = ["data/backgrounds/" + x for x in os.listdir("data/backgrounds/")]

#list training_act
with open("training_act.txt","r") as f:
    lines = f.readlines()[:-1] #Exclude last empty line
    train_act = [line.strip() for line in lines]

#list training_neg
with open("training_neg.txt","r") as f:
    lines = f.readlines()[:-1] #Exclude last empty line
    train_neg = [line.strip() for line in lines]

#list validation_act
with open("validation_act.txt","r") as f:
    lines = f.readlines()[:-1] #Exclude last empty line
    valid_act = [line.strip() for line in lines]

#list validation_neg
with open("validation_neg.txt","r") as f:
    lines = f.readlines()[:-1] #Exclude last empty line
    valid_neg = [line.strip() for line in lines]


#Make dataloaders for training and validation
dataloader_training = datagenerator.DataGenerator(backgrounds,train_act, train_neg, samplerate=8000, **params)
dataloader_validation = datagenerator.DataGenerator(backgrounds,valid_act, valid_neg, samplerate=8000, **params)



Tx = 625
n_freq = 128

model = model.model(input_shape = (Tx, n_freq))

print(model.summary())
model = multi_gpu_model(model)
print(model.summary())



opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


for i in range(100):
    model.fit_generator(dataloader_training, epochs = 20, use_multiprocessing=True, workers=12,
                        max_queue_size=1000, validation_data=dataloader_validation)

    loss, acc = model.evaluate_generator(dataloader_validation)
    #model.fit(X, Y, batch_size = 10, epochs=10)
    #loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)

    model.save('trains/trained_model_{}.h5'.format(i))