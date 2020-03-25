import numpy as np
import utils

import os

import datagenerator

import model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras.models import load_model

model = load_model("trains/saved-model-40-0.044.hdf5")

print("Loaded model")

model.summary()

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


# Parameters
params = {'dim': (2499,128), #Dims = (timesteps, frequency bins)
          'batch_size': 1,
          'n_classes': 1, #Only detect one type of word
          'Ty': 622, #Number of output bins
          'shuffle': True}

dataloader_validation = datagenerator.DataGenerator(backgrounds,valid_act, valid_neg, samplerate=32000, **params)





#print(model.evaluate_generator(dataloader_validation))

print("Evaluated model")

sample = dataloader_validation.__getitem__(1)

result = model.predict(sample[0])


print(sample[1][0])
utils.plt_values(sample[1][0],"Golden model")
utils.plt_values(result[0],"result")
