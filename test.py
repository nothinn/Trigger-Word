import numpy as np
import utils

import os

import datagenerator

import keras

import model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import load_model


from keras import metrics

from sklearn.utils import class_weight

from datetime import datetime

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


####ONLY USE NEEDED RAM
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)sess = tf.Session(config=config)set_session(sess)  # set this TensorFlow session as the default session for Keras
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#####



fft = datagenerator.get_fft(np.sin(np.linspace(-np.pi*10,np.pi, 128)))

words = ["bed","bird","cat","dog","happy","marvin","nine"]
# Parameters
params = {'dim': (549,101), #Dims = (timesteps, frequency bins)
          'batch_size': 64,
          'num_classes': len(words),
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



#train_act = train_act[0:500]
#train_neg = train_neg[0:500]

valid_act = train_act[0:50]
valid_neg = train_neg[0:50]





#Make dataloaders for training and validation
dataloader_training = datagenerator.DataGenerator(words=words, samplerate=22050, path_to_words="data/training/words/", **params)
dataloader_validation = datagenerator.DataGenerator(words=words, samplerate=22050, path_to_words="data/validation/words/", **params)

#dataloader_validation = datagenerator.DataGenerator(backgrounds,valid_act, valid_neg, samplerate=44100, **params)



Tx = 549
n_freq = 101

model = model.model(input_shape = (Tx, n_freq), num_classes = len(words))

print(model.summary())




#input("Waiting for input after showing summary")
#model = multi_gpu_model(model)
#print(model.summary())

#model = load_model('./Keras-Trigger-Word/models/tr_model.h5')
#print(model.summary())




opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"],sample_weight_mode = "temporal")


logdir = "logs/scalars/" + datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

filepath = "trains/saved-model-{epoch:02d}-{loss:.3f}.hdf5"


saveModel_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)




callbacks = [tensorboard_callback, saveModel_callback]


weight = len(train_neg)/len(train_act)

#class_weight = {0:1, 1:weight}

    #sample = dataloader_validation.__getitem__(1)

    #print(sample[1].shape)
    #result = model.predict(sample[0])
    #for(ingoing, outgoing) in zip(sample[1], result):
        #print(ingoing, outgoing)


history = model.fit_generator(dataloader_training, epochs = 80, use_multiprocessing=True, workers=24,#class_weight = class_weight,
                        max_queue_size=1000, callbacks=callbacks, validation_data=dataloader_validation)#,validation_data=dataloader_validation)
    #, validation_data=dataloader_validation
print("History from training:")
print(history.history.keys())
    #utils.plt_history(history, str(i))
    #loss, acc = model.evaluate_generator(dataloader_validation,use_multiprocessing=True, workers=6)
    #model.fit(X, Y, batch_size = 10, epochs=10)
    #loss, acc = model.evaluate(X_dev, Y_dev)
    #print("Dev set accuracy = ", acc)



