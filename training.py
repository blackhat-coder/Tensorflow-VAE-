
from numpy.lib.function_base import gradient
import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils import LATENT_DIM
from utils import NO_EPOCHS
from utils import BATCH_SIZE
from utils import LEARNING_RATE

from models.model_def import VAE
from models.model_def import Encoder
from models.model_def import Decoder

optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE)
keras_loss = tf.keras.losses.BinaryCrossentropy()


# Refactor Network_loss into class Network_loss
def Network_loss(mean, log_var, alpha=1, beta=1):

    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 28, 28))
        y_true = tf.cast(y_true, tf.float32)

        # assert y_true.dtype = tf.float32, "use tf.cast() to cast y_true to type tf.float32"

        latent_loss = 1 + log_var - tf.square(mean) - tf.exp(log_var)
        latent_loss = -0.5 + tf.reduce_sum(latent_loss, 1)

        recons_loss = y_true * tf.math.log(1e-10+y_pred) + (1 - y_true) * tf.math.log(1e-10 + 1 - y_pred)
        recons_loss = -tf.reduce_sum(recons_loss)

        network_loss = alpha * recons_loss + beta * latent_loss
        network_loss = tf.reduce_mean(network_loss)

        return network_loss
    
    return loss

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))

x_train = x_train/255.0


x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

x_train = x_train[:1000]
y_train = y_train[:1000]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

def train():

    model = VAE()

    for epoch in range(NO_EPOCHS):

        for step, (batch_x, batch_y) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                (predictions, mean, lg_var) = model(batch_x)

                loss = Network_loss(mean, lg_var)
                loss_ = loss(batch_x, predictions)

                # loss_ = keras_loss(batch_x, predictions)
            
            gradients = tape.gradient(loss_, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(f"Epoch: {epoch}\nloss: {loss_}")
        # Write Checkpoint at epoch level or at batch level


def test_loss():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cv2
    x_t = x_train[4:5]

    t_model = VAE()
    t_y, m, v = t_model(x_t)

    print("pred shape")
    print(t_y.shape)
    print("input shape")
    print(x_t.shape)

    t_y = np.reshape(t_y, [28, 28])
    x_t = np.reshape(x_t, [28, 28])


    # cv2.imwrite("x_t.jpg", x_t)
    # cv2.imwrite("t_y.jpg", t_y)
    
    print(f"loss:{ Network_loss(m, v)(x_t, t_y) } ")

    # print(np.array(x_t))
    plt.imshow(x_t)

# things to do tomorrow:
# - write training_loop and test the model
# - adjust the loss func! it's getting too large


# - get_data(get_data func)
# - write_checkpoints
# - write logs for tensor_board

if __name__ == "__main__":
    print("Starting tarining.py")
    # train()
    test_loss()
    print("session ended")
    # print("hello world!")