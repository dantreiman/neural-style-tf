"""Regularization transformations borrowed from lucid/optvis/transform.py"""
import numpy as np
import tensorflow as tf
import uuid


def _angle2rads(angle, units):
    angle = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14 * angle / 180.
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle
    return angle


def _rand_select(xs, seed=None):
    rand_n = tf.random_uniform((), 0, len(xs), "int32", seed=seed)
    return tf.constant(xs)[rand_n]


def normalize_gradient(grad_scales=None):
    if grad_scales is not None:
        grad_scales = np.float32(grad_scales)
    op_name = "NormalizeGrad_" + str(uuid.uuid4())
    @tf.RegisterGradient(op_name)
    def _NormalizeGrad(op, grad):
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, [1, 2, 3], keepdims=True))
        if grad_scales is not None:
            grad *= grad_scales[:, None, None, None]
        return grad / grad_norm

    def inner(x):
        with x.graph.gradient_override_map({"Identity": op_name}):
            x = tf.identity(x)
        return x

    return inner


def jitter(d, seed=None):
    def inner(t_image):
        t_image = tf.convert_to_tensor(t_image, preferred_dtype=tf.float32)
        t_shp = tf.shape(t_image)
        crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
        crop = tf.random_crop(t_image, crop_shape, seed=seed)
        shp = t_image.get_shape().as_list()
        mid_shp_changed = [
            shp[-3] - d if shp[-3] is not None else None,
            shp[-2] - d if shp[-3] is not None else None,
        ]
        crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
        return crop

    return inner


def pad(w, mode="REFLECT", constant_value=0.5):
    def inner(t_image):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value
        return tf.pad(
            t_image,
            [(0, 0), (w, w), (w, w), (0, 0)],
            mode=mode,
            constant_values=constant_value_,
        )

    return inner


def random_rotate(angles, units="degrees", seed=None):
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        angle = _rand_select(angles, seed=seed)
        angle = _angle2rads(angle, units)
        return tf.contrib.image.rotate(t, angle)

    return inner


def random_scale(scales, seed=None):
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        scale = _rand_select(scales, seed=seed)
        shp = tf.shape(t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
        return tf.image.resize_bilinear(t, scale_shape)

    return inner


def make_point_grid(ny, nx, h, w, margin_x=.2, margin_y=.2):
    x = np.linspace(0.2, 0.8, ny)
    y = np.linspace(0.2, 0.8, nx)
    xv, yv = np.meshgrid(x, y)
    return np.dstack([yv * h, xv * w]).reshape([nx*ny, 2])


def random_warp(ny, nx, h, w, random_scale=64, seed=None):
    def inner(t):
        source_points = make_point_grid(ny, nx, h, w)
        source_points_t = tf.constant(source_points, dtype=tf.float32)
        dest_points_t = source_points_t + tf.random_normal(source_points.shape, seed=seed) * random_scale
        n = tf.shape(t)[0]
        warped_image_t = tf.contrib.image.sparse_image_warp(
            t,
            tf.tile(tf.expand_dims(source_points_t, 0), [n, 1, 1]),
            tf.tile(tf.expand_dims(dest_points_t, 0), [n, 1, 1]),
            interpolation_order=2,
            regularization_weight=0.0,
            num_boundary_points=2
        )
        return warped_image_t
    return inner


RANDOM_SEED = 47


standard_transforms = [
    pad(12, mode="constant", constant_value=.5),
    jitter(8, seed=RANDOM_SEED),
    random_scale([1 + (i - 1) / 50. for i in range(3)], seed=RANDOM_SEED),
    random_rotate(list(range(-10, 11)) + 5 * [0], seed=RANDOM_SEED),
    jitter(4, seed=RANDOM_SEED),
]


translate_only = [
    pad(4),
    jitter(8, seed=RANDOM_SEED)
]
