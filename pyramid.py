import numpy as np
import tensorflow as tf


def gaussian(x, n_c=3):
    gaussian_5x5 = tf.constant(np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 25, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 25, 16, 4],
        [1, 4, 6, 4, 1],
    ], dtype=np.float32) / 256.0)
    gaussian_weights = tf.reshape(gaussian_5x5, (5, 5, 1, 1))
    gaussian5x5_rgb = tf.concat([
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 0], axis=3), gaussian_weights, strides=[1, 2, 2, 1], padding='VALID'),
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 1], axis=3), gaussian_weights, strides=[1, 2, 2, 1], padding='VALID'),
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 2], axis=3), gaussian_weights, strides=[1, 2, 2, 1], padding='VALID')
    ], axis=3)
    return gaussian5x5_rgb


def laplacian(x, n_c=3):
    laplacian_operator = tf.constant(np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ], dtype=np.float32) / 4.0)
    lap_weights = tf.reshape(laplacian_operator, (3, 3, 1, 1))
    laplacian3x3_rgb = tf.concat([
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 0], axis=3), lap_weights, strides=[1, 2, 2, 1], padding='VALID'),
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 1], axis=3), lap_weights, strides=[1, 2, 2, 1], padding='VALID'),
        tf.nn.conv2d(tf.expand_dims(x[:, :, :, 2], axis=3), lap_weights, strides=[1, 2, 2, 1], padding='VALID')
    ], axis=3)
    return laplacian3x3_rgb


def bilinear(x, n_c=3, factor=0.75):
    new_size = tf.to_int32(tf.shape(x)[1:3] * factor)
    return tf.image.resize_bilinear(x, new_size)
