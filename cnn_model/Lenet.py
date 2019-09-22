# -*- coding: utf-8 -*-

from keras import Sequential
from keras.backend import image_data_format
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

img_width, img_height = 224, 224

if image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# 模型定义
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=input_shape, activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()
