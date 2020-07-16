# Disable all TensorFlow debug output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.random import truncated_normal

import numpy as np

from datetime import datetime

# Load MNIST dataset
# The both variables are pairs of images and their respective labels each
mnist = keras.datasets.mnist
train_data, test_data = mnist.load_data()


def prepare(d):
    '''Cast to float, normalize and reshape
       from 28x28 2D images to a vector of 784'''
    return (d[0].astype(np.float32).reshape([-1, 784]) / 255.0,
            d[1].astype(np.int32))


# (Hyper)Parameters for training; these can be adjusted
BATCH_SIZE = 32
TRAIN_STEPS = train_data[0].shape[0] // BATCH_SIZE  # = 1875 (default)
EPOCHS = 1
LEARNING_RATE = 0.1

# PREPROCESS
train_data = prepare(train_data)
test_data = prepare(test_data)
# Construct `Datasets`. This is optional but makes several things simple:
# We can easily get hold of an infinite iterator that works even
# over epoch boundaries
train_ds = Dataset.from_tensor_slices(train_data).repeat().shuffle(
    10000).batch(BATCH_SIZE)
test_ds = Dataset.from_tensor_slices(train_data).repeat().shuffle(10000).batch(
    BATCH_SIZE * 64)  # Larger batches to make evaluation simpler
# Batch iterator, for custom training loop
train_iter = iter(train_ds)
test_iter = iter(test_ds)

# Optional, setup logging for TensorBoard
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                   histogram_freq=1)

#
# Low-level API
#

# Weights for hidden and output layer
W = [
    tf.Variable(truncated_normal([784, 128], stddev=0.1)),
    tf.Variable(truncated_normal([128, 10], stddev=0.1))
]
# Biases
b = [
    tf.Variable(truncated_normal([128], stddev=0.1)),
    tf.Variable(truncated_normal([10], stddev=0.1))
]

# We will reuse the Stochastic Gradient Descent optimizer in the
# high-level approach later
sgd = keras.optimizers.SGD(learning_rate=LEARNING_RATE)


# Note the use of `tf.function`. This instructs TF to construct a graph from
# the function, which can help with performance. However, such annotated functions
# can behave different than regular Python functions if not used properly
@tf.function
def step(images, labels, train=True):
    '''Perform a (training) step on the provided data
     returning the loss and probability'''
    with tf.GradientTape() as tape:
        # Compute logits for the hidden and output layer
        logits_hidden = tf.nn.relu(tf.matmul(images, W[0]) + b[0])
        logits = tf.matmul(logits_hidden, W[1]) + b[1]

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=labels))
        probability = tf.nn.softmax(logits)

    if train:
        # Compute and apply the gradient
        # Alternatively, naive manual optmizer with
        # `W[0].assign_sub(learning_rate * grads[0])`
        grads = tape.gradient(loss, W + b)
        sgd.apply_gradients(zip(grads, W + b))

    return loss, probability


@tf.function
def predict(image):
    # Similar to `step()` but without providing labels
    logits_hidden = tf.nn.relu(tf.matmul(image, W[0]) + b[0])
    logits = tf.matmul(logits_hidden, W[1]) + b[1]
    probability = tf.nn.softmax(logits)
    # The label with the highest probability
    # our best guess
    return tf.argmax(probability, axis=1)


@tf.function
def measure_accuracy():
    '''Measure the models accuracy on a batch from
     the test dataset'''
    test_images, test_label = next(test_iter)
    # Perform a step on the test data without
    # training the model
    loss, probability = step(test_images, test_label, train=False)
    # Find the most likely label for each image
    predictions = tf.argmax(probability, axis=1, output_type=tf.int32)
    # Compare the guessed with the actual labels
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, test_label), tf.float32))
    return loss, accuracy


def log(epoch, epochs, loss, accuracy):
    # Print out to the console
    print(
        f"Epoch {epoch}/{epochs}: loss={loss:.4f}" \
        f" - accuracy={accuracy:.4f}"
    )
    # Log to the summary writer for the data to be
    # accessible through TensorBoard
    tf.summary.scalar('epoch_accuracy', data=accuracy, step=epoch)
    tf.summary.scalar('epoch_loss', data=loss, step=epoch)


def training_loop(epochs):
    for i in range(TRAIN_STEPS * epochs):
        # Fetch a new batch of examples and train on them
        image_batch, label_batch = next(train_iter)
        step(image_batch, label_batch, train=True)

        if i % TRAIN_STEPS == 0 and i / TRAIN_STEPS >= 1:
            # Measure performance (accuracy) each epoch
            loss, accuracy = measure_accuracy()
            log(i // TRAIN_STEPS, epochs, loss, accuracy)

    loss, accuracy = measure_accuracy()
    log(epochs, epochs, loss, accuracy)


print("Training low-level")
training_loop(EPOCHS)
file_writer.close()  # Finish logging

#
# High-level (tf.keras) API
#

# Define the model
model = keras.Sequential([
    keras.Input(shape=(784, )),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

# Specify optimizer and loss function
model.compile(optimizer=sgd,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Finally, we train the model
print("\nTraining high-level")
model.fit(train_ds,
          epochs=EPOCHS,
          steps_per_epoch=TRAIN_STEPS,
          callbacks=[tensorboard_callback])

# PREDICTING DIGITS
#
# Specific images can be classified with the
# following code. Here we use the first image
# from `train_data`, depicting a five
five = train_data[0][0].reshape(1, 784)
guess_low_level = predict(five).numpy()[0]
guess_high_level = tf.argmax(model.predict(five), axis=1).numpy()[0]
print(
    f"Predictions:\nLow-level-> {guess_low_level}, " \
    f"High-level-> {guess_high_level}"
)
