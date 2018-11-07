# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:36:09 2018

@author: Tze Hei
"""

import keras

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10 
import matplotlib.pyplot as plt 

(train_images, train_classes), (x_test, y_test) = cifar10.load_data()

train_images = train_images / 255.0
x_test = x_test / 255.0

train_classes = to_categorical(train_classes)
y_test = to_categorical(y_test)

def create_model():
    model = Sequential()
    
    model.add(Conv2D(100, (2,2), strides = (1,1), activation = 'relu', input_shape = (32,32,3)))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Conv2D(100, (2,2), strides = (1,1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    
    model.fit(train_images, train_classes, epochs = 2)
    
    model.save('cifar10test.h5')
    
    return(model)
    
    
finished_model = load_model('cifar10test.h5')

plt.imshow(x_test[0])
imtype = finished_model.predict(x_test)

print(imtype)

scores = model.evaluate(x_test, y_test)