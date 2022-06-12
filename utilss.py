
import os

BASE_DIR = os.getcwd()

MODEL_PATH = f'{BASE_DIR}\\models'
CHECKPOINT_PATH = f'{MODEL_PATH}\\checkpoints'
DATA_PATH = f'{BASE_DIR}\\data'
LOG_DIR = f'{BASE_DIR}\\logs'

INPUT_SIZE = 28
LATENT_DIM = 20
BATCH_SIZE = 32
NO_EPOCHS = 3
LEARNING_RATE = 1e-3
optimizer = "Adam"
