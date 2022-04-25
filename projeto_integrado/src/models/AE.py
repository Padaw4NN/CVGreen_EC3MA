"""
@author: Alexander Cardoso
"""

import keras
from keras import layers

class Autoencoder(keras.Model):
    def __init__(self, latent_dim, _shape_):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_size = _shape_

        self.encoder = keras.models.Sequential(
            [
                layers.Flatten(),
                layers.Dense(units = self.latent_dim, activation = "relu")
            ]
        )
        self.decoder = keras.models.Sequential(
            [
                layers.Dense(units = self.output_size, activation = "relu"),
                layers.Reshape((300, 300))
            ]
        )
    def get_autoencoder(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
