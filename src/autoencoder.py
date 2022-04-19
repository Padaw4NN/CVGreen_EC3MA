"""
@author: alexander cardoso
"""
import keras
from keras.layers import Dense
from keras.models import Model
from keras.layers import GaussianNoise

def decoder(enc_dimen, _shape_):

    input_img = keras.Input(shape=(_shape_,))
    
    encoded = Dense(units=enc_dimen, activation="relu")(input_img)
    encoded = Dense(units=enc_dimen, activation="relu")(encoded)
    decoded = Dense(units=_shape_, activation="sigmoid")
    
    return (encoded, decoded)

def running_autoencoder(enc_dimen, _shape_, decoded, x_train, x_test, b_size=5, eps=50):
    
    input_img = keras.Input(shape=(_shape_,))
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, decoder(1))
    decoder_layer = autoencoder.layers[-1]

    encoded_input = keras.Input(shape=(enc_dimen,))
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    x_gaussian_train = GaussianNoise(stddev=0.3)(x_train)

    autoencoder.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    autoencoder.fit(
        x_train, x_gaussian_train, batch_size=b_size, epochs=eps,
        shuffle=True, validation_data=(x_test, x_test))

