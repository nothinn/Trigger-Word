#Chooses a word as trigger and listens to all those files to verify
#Also makes a list of randomly chosen words, not including trigger word.

import random
import numpy as np
import os

#Check that word exists
wordNotFound = True
lines = []
while wordNotFound:
    word = input("What is your trigger word? ")
    word.lower()
    try:
        for file in os.listdir("data/words/" + word):
            lines.append("data/words/" + word + "/" + file)
        print("Number of activation sounds is {}".format(len(lines)))
        wordNotFound = False
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print("Word not found. Choose an available one")
        exit

#Randomly distribute trigger words in a training and validation set
share = input("How large a share should be in the training set (0.1 - 0.99): ")

trains = int(float(share) * len(lines))

random.shuffle(lines)
training_act = lines[0:trains] #Activations
validation_act = lines[trains:]#Activations

#make a list of all available words
dirs = os.listdir("data/words/")

dirs.remove(word) #Don't include the trigger word

negatives = []

for dir in dirs:
    try:
        for file in os.listdir("data/words/" + dir):
            negatives.append("data/words/" + dir + "/" + file)
    except:
        pass


#Randomly shuffle negatives
random.shuffle(negatives)


training_neg = negatives[0:int(float(share)*len(negatives))]
validation_neg = negatives[int(float(share)*len(negatives)):]



#Save four files, {training/validation}_{act/neg}

with open("training_act.txt","w") as f:
    for line in training_act:
        f.write(line)
        f.write("\n")
        

with open("training_neg.txt","w") as f:
    for line in training_neg:
        f.write(line)
        f.write("\n")


with open("validation_act.txt","w") as f:
    for line in validation_act:
        f.write(line)
        f.write("\n")

with open("validation_neg.txt","w") as f:
    for line in validation_neg:
        f.write(line)
        f.write("\n")

