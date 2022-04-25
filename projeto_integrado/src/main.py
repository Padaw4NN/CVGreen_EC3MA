"""
@author: Alexander Cardoso
"""

import os
import IDG, gets
import tensorflow as tf

#from models import AE
from models import CNNM

def main(epochs, steps_per_epoch, validation_steps):

    train_dir = gets.GetDirs.get_train_dir()
    test_dir = gets.GetDirs.get_test_dir()

    cnn_model = CNNM.ModelClassifier.model__()

    train_data_gen = IDG.ImgDataGen.train_data_generator(train_dir = train_dir)
    test_data_gen = IDG.ImgDataGen.train_data_generator(train_dir = test_dir)

    metrics = gets.CompileFunctions.get_metrics() # sparse categorical accuracy
    loss = gets.CompileFunctions.get_loss_function() # sparse categorical crossentropy

    cnn_model.compile(
        optimizer = tf.optimizers.Adam(),
        loss =  loss,
        metrics = [metrics]
    )
    cnn_model.fit(
        x = train_data_gen,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,

        validation_data = test_data_gen,
        validation_steps = validation_steps
    )

if __name__ == "__main__":
    os.system("clear")
    main(epochs=5, steps_per_epoch=10, validation_steps=10)
