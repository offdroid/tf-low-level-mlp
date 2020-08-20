# Disable all TensorFlow debug output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

import numpy as np

import dataset
import lowlevel
import keras

from datetime import datetime

# (Hyper)Parameters for training; these can be adjusted
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 0.1

ds = dataset.dataset(BATCH_SIZE)
TRAIN_STEPS = dataset.size() // BATCH_SIZE  # = 1875 (default)

# Optional, setup logging for TensorBoard
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

# Low-level API
low = lowlevel.Model(ds, LEARNING_RATE, TRAIN_STEPS)
print("Training low-level")
low.train(EPOCHS)
file_writer.close()  # Finish logging
print()

# High-level (tf.keras) API
high = keras.Model(ds, LEARNING_RATE, TRAIN_STEPS, tensorboard_callback)
print("Training high-level")
high.train(EPOCHS)
print()

# Classify an image
five = tf.reshape(
    tf.image.decode_png(tf.io.read_file('../five.png'), channels=1), [1, 784])
five = tf.divide(tf.dtypes.cast(five, tf.float32), 255)
guess_low_level = low.predict(five)
guess_high_level = high.predict(five)
print(
    f"Predictions:\nLow-level-> {guess_low_level}, " \
    f"High-level-> {guess_high_level}"
)
