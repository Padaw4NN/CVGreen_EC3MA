"""
@author: Alexander Cardoso
"""

import numpy as np
import keras.layers as layers
import keras.models as models
from keras.preprocessing.image import ImageDataGenerator

def building_model(in_shape=(64, 64, 3)):
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(
        filters=32,
        activation="relu",
        kernel_size=(3, 3),
        input_shape=(in_shape)
    ))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(
        filters=32,
        activation="relu",
        kernel_size=(3, 3),
        input_shape=(in_shape)
    ))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=128, activation="relu"))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Dense(units=128, activation="relu"))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Dense(units=1, activation="sigmoid"))

    return model

model = building_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=7,
    shear_range=0.2,
    height_shift_range=0.07,
    zoom_range=0.2
)

test_generator = ImageDataGenerator(rescale=1./255)























