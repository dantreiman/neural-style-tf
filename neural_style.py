import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import struct
import errno
import time
import transform  # Borrowed from /data/notebooks/deepdream/transform.py
import cv2
import os
import pyramid
import vgg19


def parse_args():
    desc = "TensorFlow implementation of 'A Neural Algorithm for Artistic Style'"
    parser = argparse.ArgumentParser(description=desc)

    # options for single image
    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag indicating if statements should be printed to the console.')

    parser.add_argument('--img_name', type=str,
                        default='result',
                        help='Filename of the output image.')

    parser.add_argument('--style_imgs', nargs='+', type=str,
                        help='Filenames of the style images (example: starry-night.jpg)',
                        required=True)

    parser.add_argument('--style_imgs_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Interpolation weights of each of the style images. (example: 0.5 0.5)')

    parser.add_argument('--content_img', type=str,
                        help='Filename of the content image (example: lion.jpg)')

    parser.add_argument('--style_imgs_dir', type=str,
                        default='./styles',
                        help='Directory path to the style images. (default: %(default)s)')

    parser.add_argument('--content_img_dir', type=str,
                        default='./image_input',
                        help='Directory path to the content image. (default: %(default)s)')

    parser.add_argument('--init_img_type', type=str,
                        default='content',
                        choices=['random', 'content', 'style'],
                        help='Image used to initialize the network. (default: %(default)s)')

    parser.add_argument('--superpixel_scale', type=float,
                        default=-1,
                        help='Scale factor for rendering using superpixel method.  -1 to disable scaling.')

    parser.add_argument('--max_size', type=int,
                        default=4096,
                        help='Maximum width or height of the input images. (default: %(default)s)')

    parser.add_argument('--content_weight', type=float,
                        default=5e0,  # 5e0
                        help='Weight for the content loss function. (default: %(default)s)')

    parser.add_argument('--style_weight', type=float,
                        default=1e4,
                        help='Weight for the style loss function. (default: %(default)s)')

    parser.add_argument('--tv_weight', type=float,
                        default=1e-3,
                        help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')

    parser.add_argument('--temporal_weight', type=float,
                        default=2e2,
                        help='Weight for the temporal loss function. (default: %(default)s)')

    parser.add_argument('--content_loss_function', type=int,
                        default=1,
                        choices=[1, 2, 3],
                        help='Different constants for the content layer loss function. (default: %(default)s)')

    parser.add_argument('--content_layers', nargs='+', type=str,
                        default=['conv4_2'],
                        help='VGG19 layers used for the content image. (default: %(default)s)')

    parser.add_argument('--style_layers', nargs='+', type=str,
                        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for the style image. (default: %(default)s)')

    parser.add_argument('--content_layer_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Contributions (weights) of each content layer to loss. (default: %(default)s)')

    parser.add_argument('--style_layer_weights', nargs='+', type=float,
                        default=[0.2, 0.2, 0.2, 0.2, 0.2],
                        help='Contributions (weights) of each style layer to loss. (default: %(default)s)')

    parser.add_argument('--downsample_method', type=str, default='resize',
                        help='One of {gaussian, laplacian, bilinear}')

    parser.add_argument('--octaves', type=int, default=1,
                        help='Each octave represents an additonal level of detail in an image pyramid.')

    parser.add_argument('--original_colors', action='store_true',
                        help='Transfer the style but not the colors.')

    parser.add_argument('--color_convert_type', type=str,
                        default='yuv',
                        choices=['yuv', 'ycrcb', 'luv', 'lab'],
                        help='Color space for conversion to original colors (default: %(default)s)')

    parser.add_argument('--color_convert_time', type=str,
                        default='after',
                        choices=['after', 'before'],
                        help='Time (before or after) to convert to original colors (default: %(default)s)')

    parser.add_argument('--style_mask', action='store_true',
                        help='Transfer the style to masked regions.')

    parser.add_argument('--style_mask_imgs', nargs='+', type=str,
                        default=None,
                        help='Filenames of the style mask images (example: face_mask.png) (default: %(default)s)')

    parser.add_argument('--noise_ratio', type=float,
                        default=1.0,
                        help="Interpolation value between the content image and noise image if the network is initialized with 'random'.")

    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Seed for the random number generator. (default: %(default)s)')

    parser.add_argument('--model_weights', type=str,
                        default='imagenet-vgg-verydeep-19.mat',
                        help='Weights and biases of the VGG-19 network.')

    parser.add_argument('--pooling_type', type=str,
                        default='avg',
                        choices=['avg', 'max'],
                        help='Type of pooling in convolutional neural network. (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='/gpu:0',
                        choices=['/gpu:0', '/cpu:0'],
                        help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')

    parser.add_argument('--img_output_dir', type=str,
                        default='./image_output',
                        help='Relative or absolute directory path to output image and data.')

    # optimizations
    parser.add_argument('--optimizer', type=str,
                        default='lbfgs',
                        choices=['lbfgs', 'adam', 'adam_adaptive', 'mixed', 'gd', 'adagrad', 'nesterov'],
                        help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')

    parser.add_argument('--early_stopping', action='store_true',
                        help='Stop each frame early if loss change is below a target. Only works for ADAM.')

    parser.add_argument('--min_iterations', type=int, default=100,
                        help='Minimum number of iterations.  Used with early stopping.')

    parser.add_argument('--transforms', type=str,
                        default='none',
                        choices=['none', 'translate', 'standard'],
                        help='Applies random jitter or rotation to image to smooth out neural network artifacts. (default|recommended: %(default)s)')

    parser.add_argument('--learning_rate', type=float,
                        default=1e0,
                        help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')

    parser.add_argument('--max_iterations', type=int,
                        default=1000,
                        help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')

    parser.add_argument('--print_iterations', type=int,
                        default=50,
                        help='Number of iterations between optimizer print statements. (default: %(default)s)')

    # options for video frames
    parser.add_argument('--video', action='store_true',
                        help='Boolean flag indicating if the user is generating a video.')

    parser.add_argument('--start_frame', type=int,
                        default=1,
                        help='First frame number.')

    parser.add_argument('--end_frame', type=int,
                        default=1,
                        help='Last frame number.')

    parser.add_argument('--first_frame_type', type=str,
                        choices=['random', 'content', 'style'],
                        default='content',
                        help='Image used to initialize the network during the rendering of the first frame.')

    parser.add_argument('--init_frame_type', type=str,
                        choices=['prev_warped', 'prev', 'random', 'content', 'style'],
                        default='prev_warped',
                        help='Image used to initialize the network during the every rendering after the first frame.')

    parser.add_argument('--flow_input_dir', type=str,
                        help='Relative or absolute directory path to optical flow files. Defaults to video_input_dir if not specified.')

    parser.add_argument('--video_input_dir', type=str,
                        default='./video_input',
                        help='Relative or absolute directory path to input frames.')

    parser.add_argument('--video_output_dir', type=str,
                        default='./video_output',
                        help='Relative or absolute directory path to output frames.')

    parser.add_argument('--content_frame_frmt', type=str,
                        default='frame_{}.png',
                        help='Filename format of the input content frames.')

    parser.add_argument('--backward_optical_flow_frmt', type=str,
                        default='backward_{}_{}.flo',
                        help='Filename format of the backward optical flow files.')

    parser.add_argument('--forward_optical_flow_frmt', type=str,
                        default='forward_{}_{}.flo',
                        help='Filename format of the forward optical flow files')

    parser.add_argument('--content_weights_frmt', type=str,
                        default='reliable_{}_{}.txt',
                        help='Filename format of the optical flow consistency files.')

    parser.add_argument('--prev_frame_indices', nargs='+', type=int,
                        default=[1],
                        help='Previous frames to consider for longterm temporal consistency.')

    parser.add_argument('--first_frame_iterations', type=int,
                        default=2000,  # 2000
                        help='Maximum number of optimizer iterations of the first frame. (default: %(default)s)')

    parser.add_argument('--frame_iterations', type=int,
                        default=800,  # 800
                        help='Maximum number of optimizer iterations for each frame after the first frame. (default: %(default)s)')

    args = parser.parse_args()

    # normalize weights
    args.style_layer_weights = normalize(args.style_layer_weights)
    args.content_layer_weights = normalize(args.content_layer_weights)
    args.style_imgs_weights = normalize(args.style_imgs_weights)

    # create directories for output
    if args.video:
        maybe_make_directory(args.video_output_dir)
    else:
        maybe_make_directory(args.img_output_dir)

    return args


'''
  'a neural algorithm for artistic style' loss functions
'''


def content_layer_loss(p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if args.content_loss_function == 1:
        K = 1. / (2. * N ** 0.5 * M ** 0.5)
    elif args.content_loss_function == 2:
        K = 1. / (N * M)
    elif args.content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.square(x - p))
    return loss


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


def mask_style_layer(a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = get_mask_image(mask_img, w.value, h.value)
    mask = tf.convert_to_tensor(mask)
    tensors = []
    for _ in range(d.value):
        tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x


def sum_masked_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    masks = args.style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(stem['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


def sum_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(stem['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


'''
  'artistic style transfer for videos' loss functions
'''


def temporal_loss(x, w, c):
    """Compute temporal consistency loss.

    Args:
      x (tf.Tensor) The image.
      w (tf.Tensor) The warped frame.
      c (tf.Tensor) The content weights.
    """
    c = tf.expand_dims(c, 0)
    D = float(np.prod(x.get_shape().as_list()))
    print('temporal loss: D = %f' % D)
    loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss


def get_longterm_weights(i, j):
    c_sum = 0.
    for k in range(args.prev_frame_indices):
        if i - k > i - j:
            c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max


def sum_longterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(stem['input'].assign(input_img))
    loss = 0.
    for j in range(args.prev_frame_indices):
        prev_frame = frame - j
        w = get_prev_warped_frame(frame)
        c = get_longterm_weights(frame, prev_frame)
        loss += temporal_loss(x, w, c)
    return loss


'''
  utilities and i/o
'''


def read_image(path):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img


def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)


def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis, :, :, :]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre


def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[..., ::-1]
    return imgpost


def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0, y, x] = struct.unpack('f', f.read(4))[0]
                flow[1, y, x] = struct.unpack('f', f.read(4))[0]
    return flow


def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i - 1] = np.array(list(map(np.float32, line)))
        vals[i - 1] = list(map(lambda x: 0. if x < 255. else 1., vals[i - 1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights


def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)


def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)


class Model:
    def __init__(self):
        with tf.device(args.device):
            self.sess = tf.Session()
            self.stem = {}  # The common portion of the neural network, before image pyramid
            self.nets = []  # The networks for each layer of the pyramid, from largest to smallest
            self.update_content_ops = []
            self.content_weights = None
            self.loss = None
            self.debug_losses = {}
            self.tf_optimizer = None
            self.sc_optimizer = None
            self.train_op = None
            self.max_tf_iterations = None  # Max iterations of whichever TF optimizer wer're using.
            self.max_bfgs_iterations = None  # Max iterations of L-BFGS.
            self.set_max_iterations(args.max_iterations)

    def set_max_iterations(self, max_iterations):
        if args.optimizer == 'mixed':
            self.max_tf_iterations = 400
            self.max_bfgs_iterations = max_iterations
        else:
            self.max_tf_iterations = max_iterations
            self.max_bfgs_iterations = max_iterations

    def build_network(self, input_img):
        _, h, w, d = input_img.shape
        stem = {}
        stem['input_in'] = tf.placeholder(tf.float32, shape=(1, h, w, d))  # Used to feed images into input
        stem['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32), trainable=True)
        stem['input_assign'] = stem['input'].assign(stem['input_in'])
        stem['prev_input_in'] = tf.placeholder(tf.float32, shape=(
            1, h, w, d))  # Used to feed previous image into temporal loss function.
        stem['prev_input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32),
                                         trainable=False)  # Previous input for temporal consistency
        stem['prev_input_assign'] = stem['prev_input'].assign(stem['prev_input_in'])
        stem['global_step'] = tf.Variable(0, dtype=tf.int64, trainable=False)

        transforms = []
        if args.transforms == 'standard':
            print('Using standard transforms.')
            transforms = transform.standard_transforms
        elif args.transforms == 'translate':
            print('Using translate transform only.')
            transforms = transform.translate_only
        t_transformed = stem['input']
        for t in transforms:
            t_transformed = t(t_transformed)
        stem['input_transformed'] = t_transformed
        self.stem = stem

        net, reuse_vars = vgg19.build_network(t_transformed, args.model_weights)
        self.net = net  ## TODO: remove this once we support multiple octaves.
        self.nets.append(net)

        # Build image pyramid if more than one octave is specified.
        downsample = pyramid.gaussian
        if args.downsample_method == 'laplacian':
            downsample = pyramid.laplacian
        elif args.downsample_method == 'resize':
            downsample = pyramid.resize

        for i in range(args.octaves - 1):
            o = downsample(t_transformed)
            net = vgg19.build_network(o, args.model_weights, reuse_vars=reuse_vars)
            self.nets.append(net)


    def load(self, init_img, content_img, style_imgs):
        """Build model and load weights.  Content image is only used for computing size."""
        # setup network
        net = self.build_network(content_img)

        # style loss
        if args.style_mask:
            L_style = sum_masked_style_losses(self.sess, net, style_imgs)
        else:
            L_style = sum_style_losses(self.sess, net, style_imgs)

        # content loss
        L_content = self.setup_content_loss(content_img)

        # denoising loss
        L_tv = tf.image.total_variation(stem['input'])

        # loss weights
        alpha = args.content_weight
        beta = args.style_weight
        theta = args.tv_weight

        # total loss
        L_total = alpha * L_content
        L_total += beta * L_style
        L_total += theta * L_tv

        # self.debug_losses['content'] = L_content
        # self.debug_losses['style'] = L_style
        # self.debug_losses['tv'] = L_tv

        # video temporal loss
        # if args.video:
        # gamma      = args.temporal_weight
        # gamma = 0.0
        # L_temporal = self.setup_shortterm_temporal_loss()
        # self.debug_losses['temporal'] = L_temporal
        # L_total   += gamma * L_temporal

        # optimization algorithm
        self.loss = L_total
        self.setup_optimizer(self.loss)
        if args.optimizer in ('adam', 'adam_adaptive', 'mixed', 'gd', 'adagrad', 'nesterov'):
            self.train_op = self.tf_optimizer.minimize(self.loss, global_step=stem['global_step'])
        self.sess.run(tf.global_variables_initializer())

    def setup_shortterm_temporal_loss(self):
        c = get_content_weights(args.start_frame, args.start_frame + 1)
        # Initializes content weights to all zeros for first frame
        self.content_weights = tf.Variable(np.zeros_like(c), trainable=False)
        return temporal_loss(self.stem['input'], self.stem['prev_input'], self.content_weights)

    def update_shortterm_temporal_loss(self, frame):
        if frame is None or frame == 0 or (frame == 1 and args.start_frame == 1):
            return
        prev_frame = max(frame - 1, 0)
        w = get_prev_warped_frame(frame)
        c = get_content_weights(frame, prev_frame)
        self.sess.run(self.stem['prev_input_assign'], feed_dict={self.stem['prev_input_in']: w})
        self.sess.run(self.content_weights.assign(c))

    def setup_content_loss(self, content_img):
        net = self.net
        self.sess.run(stem['input_assign'], feed_dict={stem['input_in']: content_img})
        content_loss = 0.
        content_layers = []
        content_vars = []
        for layer_name, weight in zip(args.content_layers, args.content_layer_weights):
            layer_t = net[layer_name]
            activations = self.sess.run(layer_t)
            content_t = tf.Variable(activations, trainable=False)
            content_loss += content_layer_loss(content_t, layer_t) * weight
            content_layers.append(layer_t)
            content_vars.append(content_t)
        self.content_vars = content_vars
        self.update_content_ops = [v.assign(l) for v, l in zip(content_vars, content_layers)]
        content_loss /= float(len(args.content_layers))
        return content_loss

    def update_content_loss(self, content_img):
        self.sess.run(self.stem['input_assign'], feed_dict={self.stem['input_in']: content_img})
        self.sess.run(self.update_content_ops)

    def stylize(self, content_img, style_imgs, init_img, frame=None):
        """Do gradient descent, save style image"""
        self.update_content_loss(content_img)
        # self.update_shortterm_temporal_loss(frame)
        net = self.net
        self.sess.run(stem['input_assign'], feed_dict={stem['input_in']: init_img})
        if args.optimizer in ('adam', 'adam_adaptive', 'mixed'):
            self.minimize_with_adam(self.loss)
        if args.optimizer in ('lbfgs', 'mixed'):
            self.minimize_with_lbfgs()
        if args.optimizer == 'gd':
            self.minimize_with_gd(self.loss)
        if args.optimizer == 'adagrad':
            self.minimize_with_adagrad(self.loss)
        if args.optimizer == 'nesterov':
            self.minimize_with_nesterov(self.loss)
        output_img = self.sess.run(stem['input'])

        if args.original_colors:
            output_img = convert_to_original_colors(np.copy(content_img), output_img)

        if args.video:
            write_video_output(frame, output_img)
        else:
            write_image_output(output_img, content_img, style_imgs, init_img)

    def minimize_with_lbfgs(self):
        if args.verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
        self.sc_optimizer.minimize(self.sess)

    def should_stop_early(self, loss_history):
        if len(loss_history) < args.min_iterations:
            return False
        y2 = loss_history[-1]
        y1 = loss_history[-2]
        pct_change = ((y2 - y1) / y1) * 100
        # Stop early if the loss decreased and the decrease was less than 0.01%
        return pct_change > -0.01 and pct_change < 0

    def minimize_with_adam(self, loss):
        if args.verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
        iterations = 0
        loss_history = []
        while (iterations < self.max_tf_iterations):
            _, l = self.sess.run([self.train_op, loss])
            loss_history.append(l)
            # if iterations % args.print_iterations == 0:
            #  for k,v in self.debug_losses.items():
            #    print('%s: %.5f' % (k, self.sess.run(v)))
            if iterations % args.print_iterations == 0 and args.verbose:
                lr = args.learning_rate
                if 'learning_rate' in self.net:
                    lr = self.sess.run(self.stem['learning_rate'])
                print("At iterate {}\tf= {}\tlr = {}".format(iterations, l, lr))
            if args.early_stopping and self.should_stop_early(loss_history):
                print('Stopping early')
                return
            iterations += 1

    def minimize_with_gd(self, loss):
        if args.verbose: print('\nMINIMIZING LOSS USING: GRADIENT DESCENT OPTIMIZER')
        iterations = 0
        while (iterations < self.max_tf_iterations):
            self.sess.run(self.train_op)
            if iterations % args.print_iterations == 0 and args.verbose:
                lr = args.learning_rate
                if 'learning_rate' in self.net:
                    lr = self.sess.run(self.stem['learning_rate'])
                curr_loss = self.sess.run(loss)
                print("At iterate {}\tf= {}\tlr = {}".format(iterations, curr_loss, lr))
            iterations += 1

    def minimize_with_adagrad(self, loss):
        if args.verbose: print('\nMINIMIZING LOSS USING: ADAGRAD OPTIMIZER')
        iterations = 0
        while (iterations < self.max_tf_iterations):
            self.sess.run(self.train_op)
            if iterations % args.print_iterations == 0 and args.verbose:
                lr = args.learning_rate
                if 'learning_rate' in self.net:
                    lr = self.sess.run(self.stem['learning_rate'])
                curr_loss = self.sess.run(loss)
                print("At iterate {}\tf= {}\tlr = {}".format(iterations, curr_loss, lr))
            iterations += 1

    def minimize_with_nesterov(self, loss):
        if args.verbose: print('\nMINIMIZING LOSS USING: NESTEROV MOMENTUM')
        iterations = 0
        while (iterations < self.max_tf_iterations):
            self.sess.run(self.train_op)
            if iterations % args.print_iterations == 0 and args.verbose:
                lr = args.learning_rate
                if 'learning_rate' in self.net:
                    lr = self.sess.run(self.stem['learning_rate'])
                curr_loss = self.sess.run(loss)
                print("At iterate {}\tf= {}\tlr = {}".format(iterations, curr_loss, lr))
            iterations += 1

    def setup_optimizer(self, loss):
        print_iterations = args.print_iterations if args.verbose else 0
        if args.optimizer in ('lbfgs', 'mixed'):
            self.sc_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method='L-BFGS-B',
                options={'maxiter': self.max_bfgs_iterations,
                         'disp': print_iterations})
        if args.optimizer in ('adam', 'mixed'):
            self.tf_optimizer = tf.train.AdamOptimizer(args.learning_rate)
        if args.optimizer == 'gd':
            self.stem['learning_rate'] = tf.train.exponential_decay(args.learning_rate, self.stem['global_step'],
                                                                    100, 0.96, staircase=True)
            self.tf_optimizer = tf.train.GradientDescentOptimizer(self.stem['learning_rate'])
        if args.optimizer == 'adam_adaptive':
            self.stem['learning_rate'] = tf.train.exponential_decay(args.learning_rate, self.stem['global_step'],
                                                                   100, 0.96, staircase=True)
            self.tf_optimizer = tf.train.AdamOptimizer(self.stem['learning_rate'])
        if args.optimizer == 'adagrad':
            self.tf_optimizer = tf.train.AdagradOptimizer(args.learning_rate)
        if args.optimizer == 'nesterov':
            self.tf_optimizer = tf.train.MomentumOptimizer(args.learning_rate, 0.9, use_nesterov=True)


def write_video_output(frame, output_img):
    fn = args.content_frame_frmt.format(str(frame).zfill(5))
    path = os.path.join(args.video_output_dir, fn)
    write_image(path, output_img)


def write_image_output(output_img, content_img, style_imgs, init_img):
    out_dir = os.path.join(args.img_output_dir, args.img_name)
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, args.img_name + '.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_' + str(index) + '.png')
        write_image(path, style_img)
        index += 1

    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(args.img_name))
    f.write('content: {}\n'.format(args.content_img))
    index = 0
    for style_img, weight in zip(args.style_imgs, args.style_imgs_weights):
        f.write('styles[' + str(index) + ']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    if args.style_mask_imgs is not None:
        for mask in args.style_mask_imgs:
            f.write('style_masks[' + str(index) + ']: {}\n'.format(mask))
            index += 1
    f.write('init_type: {}\n'.format(args.init_img_type))
    f.write('content_weight: {}\n'.format(args.content_weight))
    f.write('style_weight: {}\n'.format(args.style_weight))
    f.write('tv_weight: {}\n'.format(args.tv_weight))
    f.write('content_layers: {}\n'.format(args.content_layers))
    f.write('style_layers: {}\n'.format(args.style_layers))
    f.write('optimizer_type: {}\n'.format(args.optimizer))
    f.write('max_iterations: {}\n'.format(args.max_iterations))
    f.write('max_image_size: {}\n'.format(args.max_size))
    f.close()


'''
  image loading and processing
'''


def get_init_image(init_type, content_img, style_imgs, frame=None):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(args.noise_ratio, content_img)
        return init_img
    # only for video frames
    elif init_type == 'prev':
        init_img = get_prev_frame(frame)
        return preprocess(init_img.astype(np.float32))
    elif init_type == 'prev_warped':
        init_img = get_prev_warped_frame(frame)
        return init_img


def get_content_frame(frame):
    fn = args.content_frame_frmt.format(str(frame).zfill(5))
    path = os.path.join(args.video_input_dir, fn)
    img = get_content_image(path)
    return img


def get_content_image(content_img_path):
    path = content_img_path
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    scale_f = args.superpixel_scale
    if scale_f > 1.0:
        print('Scaling image up by %f' % scale_f)
        img = cv2.resize(img, (int(scale_f * img.shape[1]), int(scale_f * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    h, w, d = img.shape
    mx = args.max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img


def get_style_images(content_img):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in args.style_imgs:
        path = os.path.join(args.style_imgs_dir, style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs


def get_noise_image(noise_ratio, content_img):
    np.random.seed(args.seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1. - noise_ratio) * content_img
    return img


def get_mask_image(mask_img, width, height):
    path = os.path.join(args.content_img_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    check_image(img, path)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img


def get_prev_frame(frame):
    # previously stylized frame
    prev_frame = max(frame - 1, 0)
    fn = args.content_frame_frmt.format(str(prev_frame).zfill(5))
    path = os.path.join(args.video_output_dir, fn)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    return img


def get_flow_input_dir():
    if args.flow_input_dir:
        return args.flow_input_dir
    else:
        return args.video_input_dir


def resize_flow(flow, width, height):
    """Resizes optical flow to target size"""
    flow_img = np.transpose(flow, (1, 2, 0))
    scaled = cv2.resize(flow_img, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.transpose(scaled, (2, 0, 1))


def get_prev_warped_frame(frame):
    prev_img = get_prev_frame(frame)
    prev_frame = max(frame - 1, 0)
    # backwards flow: current frame -> previous frame
    fn = args.backward_optical_flow_frmt.format(str(frame), str(prev_frame))
    path = os.path.join(get_flow_input_dir(), fn)
    flow = read_flow_file(path)
    scale_f = args.superpixel_scale
    if scale_f > 1.0:
        print('Scaling optical flow up by %f' % scale_f)
        flow = resize_flow(flow, prev_img.shape[1], prev_img.shape[0])
        flow = flow * scale_f  # Multiplies displacement vectors by scale factor.
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    img = preprocess(warped_img)
    return img


def get_content_weights(frame, prev_frame):
    forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(get_flow_input_dir(), forward_fn)
    backward_path = os.path.join(get_flow_input_dir(), backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights  # , backward_weights


def warp_image(src, flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1, y, :] = float(y) + flow[1, y, :]
    for x in range(w):
        flow_map[0, :, x] = float(x) + flow[0, :, x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1],
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def convert_to_original_colors(content_img, stylized_img):
    content_img = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if args.color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif args.color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif args.color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif args.color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst


def render_single_image():
    model = Model()
    content_image_path = os.path.join(args.content_img_dir, args.content_img)
    content_img = get_content_image(content_image_path)
    style_imgs = get_style_images(content_img)
    print('\n---- RENDERING SINGLE IMAGE ----\n')
    init_img = get_init_image(args.init_img_type, content_img, style_imgs)
    tick = time.time()
    model.load(init_img, content_img, style_imgs)
    model.stylize(content_img, style_imgs, init_img)
    tock = time.time()
    print('Single image elapsed time: {}'.format(tock - tick))


def render_video():
    model = Model()
    needs_load = True
    for frame in range(args.start_frame, args.end_frame + 1):
        # If start_frame > 1, assume we are resuming a previously killed job.
        # TODO(dtreiman): check for existance of previous frame instead.
        assume_resume = args.start_frame > 1
        print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, args.end_frame))
        if not assume_resume and frame == args.start_frame:
            content_frame = get_content_frame(frame)
            style_imgs = get_style_images(content_frame)
            init_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame)
            model.set_max_iterations(args.first_frame_iterations)
            tick = time.time()
            if needs_load:
                model.load(init_img, content_frame, style_imgs)
                needs_load = False
            model.stylize(content_frame, style_imgs, init_img, frame)
            tock = time.time()
            print('Frame {} elapsed time: {}'.format(frame, tock - tick))
        else:
            content_frame = get_content_frame(frame)
            style_imgs = get_style_images(content_frame)
            init_img = get_init_image(args.init_frame_type, content_frame, style_imgs, frame)
            model.set_max_iterations(args.frame_iterations)
            tick = time.time()
            if needs_load:
                model.load(init_img, content_frame, style_imgs)
                needs_load = False
            model.stylize(content_frame, style_imgs, init_img, frame)
            tock = time.time()
            print('Frame {} elapsed time: {}'.format(frame, tock - tick))


def main():
    global args
    args = parse_args()
    if args.video:
        render_video()
    else:
        render_single_image()


if __name__ == '__main__':
    main()
