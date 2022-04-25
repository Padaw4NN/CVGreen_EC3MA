"""
@author: Alexander Cardoso
"""

import tensorflow as tf

class GetDirs():    
    def get_train_dir():
        return "/home/alex/Desktop/projeto_integrado/dataset/train/training_set"
    def get_test_dir():
        return "/home/alex/Desktop/projeto_integrado/dataset/train/test_set"

class CompileFunctions():
    def get_loss_function():
        return tf.losses.SparseCategoricalCrossentropy()
    def get_metrics():
        return tf.keras.metrics.SparseCategoricalAccuracy()
