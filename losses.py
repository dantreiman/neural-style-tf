"""'a neural algorithm for artistic style' loss functions"""

import tensorflow as tf


def weighted_content_loss(x, f, weights):
    """Compute temporal consistency loss.

    Args:
      x (tf.Tensor) The image.
      f (tf.Tensor) The target frame.
      weights (tf.Tensor) The content weights.
    """
    _, h, w, d = x.get_shape()
    #print('temporal loss: D = %f' % D)
    K = (h * w * d)
    loss = tf.reduce_sum(weights * tf.square(x - f)) / K
    return loss


def content_layer_loss(p, x, content_loss_function=1):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function == 1:
        K = 1. / (2. * N ** 0.5 * M ** 0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.square(x - p))
    return loss


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.square(G - A))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G
