
from numpy.lib.function_base import gradient
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
import os
import cv2

from utilss import LATENT_DIM
from utilss import NO_EPOCHS
from utilss import BATCH_SIZE
from utilss import LEARNING_RATE
from utilss import CHECKPOINT_PATH
from utilss import LOG_DIR
from utilss import DATA_PATH

from models.model_def import VAE
from models.model_def import Encoder
from models.model_def import Decoder

optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)


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

def LogsWriter():
    log_dir_name = datetime.datetime.now().strftime("%d%m%y-%H%M%S")

    train_writer = tf.summary.create_file_writer(f"{LOG_DIR}\\{log_dir_name}\\train")
    val_writer = tf.summary.create_file_writer(f"{LOG_DIR}\\{log_dir_name}\\val")

    return train_writer, val_writer

class DataPrep():
    def __init__(self, shuffle=True, buffer_size=1024):
        self.shuffle = True
        self.buffer_size = buffer_size
        self.rawimages = []
        self.processed_images = []

    def prepare_data(self, path=""):
        data_dir = DATA_PATH + path
        for filename in os.listdir(data_dir):
            self.rawimages.append(os.path.join(data_dir, filename))

        for image in self.rawimages:
            image_data = cv2.imread(image)
            self.processed_images.append(image_data)

        self.processed_images = np.array(self.processed_images)
        self.processed_images = self.processed_images / 255.0

        train_dataset = tf.data.Dataset.from_tensor_slices(self.processed_images)

        if self.shuffle == True:
            train_dataset = train_dataset.shuffle(self.buffer_size).batch(BATCH_SIZE)
        else:
            train_dataset = train_dataset.batch(BATCH_SIZE)

        return train_dataset

def train(write_logs=True):

    model = VAE()

    if write_logs == True:
        (train_writer_, val_writer_) = LogsWriter()

    for epoch in range(NO_EPOCHS):

        for step, (batch_x, batch_y) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                (predictions, mean, lg_var) = model(batch_x)

                loss = Network_loss(mean, lg_var)
                loss_ = loss(batch_x, predictions)
            
            gradients = tape.gradient(loss_, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        with train_writer_.as_default():
            
            tf.summary.scalar("train_loss", loss_, step=epoch)

            train_writer_.flush()

        print(f"Epoch: {epoch}\nloss: {loss_}")
        # Write Checkpoint at epoch level or at batch level

if __name__ == "__main__":
    print("Starting tarining.py")
    dd = DataPrep()
    data = dd.prepare_data("\\train_data")
    print(data.shape)
    print("session ended")