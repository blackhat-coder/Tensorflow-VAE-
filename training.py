
import tensorflow as tf
from tensorflow import keras

from utils import LATENT_DIM
from utils import NO_EPOCHS
from utils import BATCH_SIZE
from utils import LEARNING_RATE

optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE)


def KL_loss(mean, log_var, alpha=1, beta=1):

    def loss(y_true, y_pred):

        latent_loss = 1 + log_var - tf.square(mean) - tf.exp(log_var)
        latent_loss = -0.5 + tf.reduce_sum(latent_loss, 1)

        recons_loss = y_true * tf.log(1e-10+y_pred) + (1 - y_true) * tf.log(1e-10 + 1 - y_pred)
        recons_loss = -tf.reduce_sum(recons_loss)

        network_loss = alpha * recons_loss + beta * latent_loss
        network_loss = tf.reduce_mean(network_loss)

        return network_loss
    
    return loss


# things to do tomorrow:
# - write training_loop and test the model
# - get_data
# - write_checkpoints
# - write logs for tensor_board
