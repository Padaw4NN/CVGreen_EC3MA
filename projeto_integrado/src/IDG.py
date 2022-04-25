"""
@author: Alexander Cardoso
"""

from keras.preprocessing.image import ImageDataGenerator

class ImgDataGen():

    def train_data_generator(train_dir):
        train_generator = ImageDataGenerator(rescale = 1./255)
        train_generator = train_generator.flow_from_directory(
            directory = train_dir,
            target_size = (300, 300),
            batch_size = 2,
            class_mode = "binary"
        )
        return train_generator

    def test_data_generator(test_dir):
        test_generator = ImageDataGenerator(rescale = 1./255)
        test_generator = test_generator.flow_from_directory(
            directory = test_dir,
            target_size = (300, 300),
            batch_size = 2,
            class_mode = "binary"
        )
        return test_generator