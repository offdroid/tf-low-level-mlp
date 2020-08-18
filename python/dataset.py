import tensorflow as tf
import numpy as np


def size():
    return tf.keras.datasets.mnist.load_data()[0][0].shape[0]


def prepare(data):
    '''Cast to float, normalize and reshape
       from 28x28 2D images to a vector of 784'''
    return (data[0].astype(np.float32).reshape([-1, 784]) / 255.0,
            data[1].astype(np.int32))


def dataset(batch_size):
    '''Construct a training and testing dataset, split into batches and corresponding iterators'''
    # Load MNIST dataset
    # Both variables are pairs of images and their respective labels each
    mnist = tf.keras.datasets.mnist
    train_data, test_data = mnist.load_data()
    # Preprocess
    train_data = prepare(train_data)
    test_data = prepare(test_data)
    # Construct `Datasets`. This is optional but makes several things simple:
    # We can easily get hold of an infinite iterator that works even
    # over epoch boundaries
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).repeat().shuffle(
        10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(train_data).repeat().shuffle(
        10000).batch(batch_size *
                     64)  # Larger batches to make evaluation simpler
    return {
        'train': train_ds,
        'test': test_ds,
        'train_iter': iter(train_ds),
        'test_iter': iter(test_ds)
    }
