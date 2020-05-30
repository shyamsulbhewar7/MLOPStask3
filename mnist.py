
import tensorflow as tf 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.

x_train /= 255

x_test /= 255

import keras 
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization
import numpy as np
model = Sequential()
model.add(Conv2D(filters= 64, kernel_size=(7,7), strides=(), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters= 64, kernel_size=(3,3), strides=(3,3), padding='valid'))
model.add(Conv2D(filters= 128, kernel_size=(2,2), strides=(2,2), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters= 256, kernel_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters= 512, kernel_size=(2,2), strides=(2,2), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters= 1024, kernel_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout

model.add(Dropout(0.4))
model.add(Dense(2048, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout

model.add(Dropout(0.4))
model.add(Dense(1024, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout

model.add(Dropout(0.4))
model.add(Dense(17)) 
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 
g = model.fit(x_train,y_train,epochs=10)
f = model.evaluate(x_test, y_test)

s = open("accuracy.txt","w+") 
s.write(f)
s.close()
