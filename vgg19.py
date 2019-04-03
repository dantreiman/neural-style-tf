import tensorflow as tf
import numpy as np
import scipy.io


def conv_layer(layer_name, layer_input, W, verbose=True):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    if verbose:
        print('--{} | shape={} | weights_shape={}'.format(layer_name, conv.get_shape(), W.get_shape()))
    return conv


def relu_layer(layer_name, layer_input, b, verbose=True):
    relu = tf.nn.relu(layer_input + b)
    if verbose:
        print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), b.get_shape()))
    return relu


def pool_layer(layer_name, layer_input, pooling_type='avg', verbose=True):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    if verbose:
        print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool


def get_weights(vgg_layers, i, reuse_vars):
    identifier = 'weights_%d' % i
    if identifier in reuse_vars:
        return reuse_vars[identifier]
    else:
        weights = vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)
        reuse_vars[identifier] = W
        return W


def get_bias(vgg_layers, i, reuse_vars):
    identifier = 'biases_%d' % i
    if identifier in reuse_vars:
        return reuse_vars[identifier]
    else:
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        reuse_vars[identifier] = b
        return b


def build_network(input_t, weights_path, reuse_vars=None, pooling_type='avg', verbose=True):
    """Builds the VGG19 network.

    Args:
        input_t (tf.Tensor) The input image.
        weights_path (str) Path to the VGG19 model weights.
        reuse_vars (dict) Dictionary mapping strings to tf.Tensors, reusable network weights and biases.
        verbose (bool) Prints layers and dimensions if true.

    Returns (net, reuse_vars).  The network, and a dictionary of reusable network variables.
    """
    if verbose:
        print('\nBUILDING VGG-19 NETWORK')
    net = {}

    if reuse_vars is None:
        if verbose:
            print('loading model weights...')
        vgg_rawnet = scipy.io.loadmat(weights_path)
        vgg_layers = vgg_rawnet['layers'][0]
        reuse_vars = {}
    else:
        if verbose:
            print('reusing model weights')
        vgg_layers = []

    if verbose:
        print('constructing layers...')

    if verbose:
        print('LAYER GROUP 1')
    net['conv1_1'] = conv_layer('conv1_1', input_t, W=get_weights(vgg_layers, 0, reuse_vars), verbose=verbose)
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0, reuse_vars), verbose=verbose)

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2, reuse_vars), verbose=verbose)
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2, reuse_vars), verbose=verbose)

    net['pool1'] = pool_layer('pool1', net['relu1_2'], pooling_type, verbose)

    if verbose:
        print('LAYER GROUP 2')
    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5, reuse_vars), verbose=verbose)
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5, reuse_vars), verbose=verbose)

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7, reuse_vars), verbose=verbose)
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7, reuse_vars), verbose=verbose)

    net['pool2'] = pool_layer('pool2', net['relu2_2'], pooling_type, verbose)

    if verbose:
        print('LAYER GROUP 3')
    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10, reuse_vars), verbose=verbose)
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10, reuse_vars), verbose=verbose)

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12, reuse_vars), verbose=verbose)
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12, reuse_vars), verbose=verbose)

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14, reuse_vars), verbose=verbose)
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14, reuse_vars), verbose=verbose)

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16, reuse_vars), verbose=verbose)
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16, reuse_vars), verbose=verbose)

    net['pool3'] = pool_layer('pool3', net['relu3_4'], pooling_type, verbose)

    if verbose:
        print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19, reuse_vars), verbose=verbose)
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19, reuse_vars), verbose=verbose)

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21, reuse_vars), verbose=verbose)
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21, reuse_vars), verbose=verbose)

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23, reuse_vars), verbose=verbose)
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23, reuse_vars), verbose=verbose)

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25, reuse_vars), verbose=verbose)
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25, reuse_vars), verbose=verbose)

    net['pool4'] = pool_layer('pool4', net['relu4_4'], pooling_type, verbose)

    if verbose:
        print('LAYER GROUP 5')
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28, reuse_vars), verbose=verbose)
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28, reuse_vars), verbose=verbose)

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30, reuse_vars), verbose=verbose)
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30, reuse_vars), verbose=verbose)

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32, reuse_vars), verbose=verbose)
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32, reuse_vars), verbose=verbose)

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34, reuse_vars), verbose=verbose)
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34, reuse_vars), verbose=verbose)

    net['pool5'] = pool_layer('pool5', net['relu5_4'], pooling_type, verbose)
    return net, reuse_vars
