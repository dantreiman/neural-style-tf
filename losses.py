"""'a neural algorithm for artistic style' loss functions"""

import tensorflow as tf


def weighted_content_loss(x, f, weights):
    """Compute temporal consistency loss.

    Args:
      x (tf.Tensor) The image.
      f (tf.Tensor) The target frame.
      weights (tf.Tensor) The content weights.
    """
    #print('temporal loss: D = %f' % D)
    loss = tf.reduce_sum(weights * tf.square(x - f)) / (tf.reduce_sum(weights) + 1.0)
    return loss


def content_layer_loss(p, x, content_loss_function=1):
    p_shape = tf.shape(p)
    h, w, d = (p_shape[1], p_shape[2], p_shape[3])
    M = tf.cast(h * w, tf.float32)
    N = tf.cast(d, tf.float32)
    if content_loss_function == 1:
        K = 2. * N ** 0.5 * M ** 0.5
    elif content_loss_function == 2:
        K = N * M
    elif content_loss_function == 3:
        K = 2.
    loss = (1. / K) * tf.reduce_sum(tf.square(x - p))
    return loss


def style_layer_loss(a, x):
    a_shape = tf.shape(a)
    h, w, d = (a_shape[1], a_shape[2], a_shape[3])
    M = h * w
    N = d
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    K = tf.cast(4 * N ** 2 * M ** 2, tf.float32)
    loss = (1. / K) * tf.reduce_sum(tf.square(G - A))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, [area, depth])
    G = tf.matmul(tf.transpose(F), F)
    return G
