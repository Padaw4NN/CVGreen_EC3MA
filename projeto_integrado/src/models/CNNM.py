"""
@author: Alexander Cardoso
"""

import keras
from keras import layers

class ModelClassifier():
    def model__():
        model = keras.models.Sequential(
            [
                layers.Conv2D(
                    filters = 32,
                    kernel_size = (3, 3),
                    activation = "relu",
                    input_shape = (300, 300, 3)
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size = (2, 2)),

                layers.Conv2D(
                    filters = 64,
                    kernel_size = (3, 3),
                    activation = "relu"
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size = (2, 2)),
                
                layers.Flatten(),
                
                layers.Dense(units = 512, activation = "relu"),
                layers.Dense(units = 1, activation = "sigmoid")
            ]
        )
        return model
