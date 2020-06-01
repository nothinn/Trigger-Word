import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop

# Create sample model
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax', name='pred'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Make a quantized TF Lite version
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
#Save the converted model
open("converted_model.tflite", "wb").write(quantized_model)