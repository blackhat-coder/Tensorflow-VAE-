
import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils.py import INPUT_SIZE
from utils.py import LATENT_DIM


class Encoder(keras.Model):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.input_layer = keras.layers.Input(shape=(INPUT_SIZE))
        
        self.conv_layer_one = keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv_layer_two = keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.pool_one = keras.layers.MaxPool2D(2, 2)

        self.conv_layer_three = keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_layer_four = keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool_two = keras.layers.MaxPool2D(2, 2)

    
    def __call__(self, inputs):

        inputs = self.input_layer()(inputs)
        inputs = self.conv_layer_one()(inputs)
        inputs = self.conv_layer_two()(inputs)
        inputs = self.pool_one()(inputs)

        inputs = self.conv_layer_three()(inputs)
        inputs = self.conv_layer_four()(inputs)
        inputs = self.pool_two()()(inputs)

        flatten = keras.layers.Flatten()(inputs)

        mean = keras.layers.Dense(LATENT_DIM)(flatten)
        mu_ = keras.layers.Dense(LATENT_DIM)(flatten)

        model = keras.models.Model( 
            inputs=[inputs], 
            outputs=[mean, mu_]
            )
        
        return model(inputs)