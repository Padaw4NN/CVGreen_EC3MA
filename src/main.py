"""
@author: alexander cardoso
"""
import build
import autoencoder
import numpy as np

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator


"""
data_generator = ImageDataGenerator(
    rescale=1./255, validation_split=0.10)

BATCH_SIZE = 5
path = "/home/alex/Desktop/ProjetoIntegrado/TreeTrunks/new/train_set/"

train_generator = data_generator.flow_from_directory(
    directory=path, target_size=(224, 224), batch_size=BATCH_SIZE,
    shuffle=True, class_mode="categorical", seed=4231, subset="training")

valid_generator = data_generator.flow_from_directory(
    directory=path, target_size=(224, 224), batch_size=BATCH_SIZE,
    shuffle=True, class_mode="categorical", seed=4231, subset="validation")
"""


"""  TESTING USING MNIST DATASET  """

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


def main():
    pass


if __name__ == "__main__":
    main()
