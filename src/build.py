"""
@author: alexander cardoso
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

def building_model(_shape_):

    model = Sequential()
    
    model.add(Conv2D(filters=64), activation="relu", kernel_size=(3, 3), input_shape=(_shape_))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=128), activation="relu", kernel_size=(3, 3))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=256), activation="relu", kernel_size=(3, 3))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=512), activation="relu", kernel_size=(3, 3))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    
    model.add(Flatten())
    
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    
    return model
