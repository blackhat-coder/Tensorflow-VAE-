
import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils import INPUT_SIZE
from utils import LATENT_DIM


class Encoder(keras.Model):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.input_layer = keras.layers.Input(shape=(INPUT_SIZE))
        
        self.conv_layer_one = keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv_layer_two = keras.layers.Conv2D(64, 3, padding="same", activation="relu") #shape = (28, 28, 64)
        self.pool_one = keras.layers.MaxPool2D(2, 2) #shape = (14, 14, 64)

        self.conv_layer_three = keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_layer_four = keras.layers.Conv2D(128, 3, padding="same", activation="relu") #shape = (14, 14, 128)
        self.pool_two = keras.layers.MaxPool2D(2, 2) #shape = (7, 7, 128)

    
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

        enc_model = keras.models.Model( 
            inputs=[inputs], 
            outputs=[mean, mu_]
            )
        
        return enc_model(inputs)
    
    
    def get_config(self):
        return super().get_config()


class Decoder(keras.Model):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.input_layer = keras.layers.Input(shape=(LATENT_DIM) )
        self.dense_layer = keras.layers.Dense(7 * 7 * 128)
        self.reshape_layer = keras.layers.Reshape((7, 7, 128))
        

        self.conv_transpose_one = keras.layers.Conv2DTranspose(128, kernel_size=3, padding="valid", activation="relu") #shape = (14, 14, 128)
        self.conv_transpose_two = keras.layers.Conv2DTranspose(128, kernel_size=3, padding="same", activation="relu")

        self.conv_transpose_three = keras.layers.Conv2DTranspose(64, kernel_size=3, padding="valid", activation="relu") #shape = (28, 28, 64)
        self.conv_transpose_four = keras.layers.Conv2DTranspose(64, kernel_size=3, padding="same", activation="relu")

        self.conv_transpose_five = keras.layers.Conv2DTranspose(1, kernel_size=3, padding="same", activation="relu") #shape = (28, 28, 1)
        self.reshape_layer_two = keras.layers.Reshape((28, 28))

    def __call__(self, inputs):
        
        dec_model = keras.models.Sequential([
            self.input_layer,
            self.dense_layer,
            self.reshape_layer,

            self.conv_transpose_one,
            self.conv_transpose_two,

            self.conv_transpose_three,
            self.conv_transpose_four,

            self.conv_transpose_five,
            self.reshape_layer_two            
        ])

        return dec_model(inputs)

    def get_config(self):
        return super().get_config()



class Sampling(keras.layers.Layer):
    
    '''
    Reparamaterization Trick
    '''

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def __call__(self, inputs):

        '''
        Args:
            inputs - A tuple containing (mean, variance)
            output - A vector of shape LATENT_DIM
        '''

        mean, log_var = inputs
        sample = tf.random.normal((LATENT_DIM, 1)) * tf.exp(log_var / 2) + mean

        return sample

class VAE(keras.Model):
    # Implement Holding the latent Vector

    def __init__(self, **kwargs):
        super(VAE, super).__init__(**kwargs)

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    
    def __call__(self, inputs):

        inputs = inputs

        mean, log_var = self.encoder()(inputs) #shape : INPUT_SIZE

        latent_vector = Sampling()((mean, log_var))

        reconstruction = self.decoder()(latent_vector)

        vae_model = keras.models.Model(inputs=[inputs], outputs=[reconstruction])

        return reconstruction