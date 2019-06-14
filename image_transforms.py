"""Geometric image transformations."""

import numpy as np
import scipy.ndimage


def zoom(image, frame_index, scale=0.0125):
    """Basic zoom: scale image up by scale, centered crop."""
    h, w = image.shape[:2]
    return scipy.ndimage.affine_transform(image, [1 - scale, 1 - scale, 1],
                                          [h * scale / 2, w * scale / 2, 0], order=3)


def unit_grid(width, height, center=None):
    """Make a grid with coordinates in the range -1 to 1

    Returns:
      (2, width, height) - The x and y values at each point in the grid, respectively.
    """
    max_dimension = float(max(width, height))
    x = np.linspace(-width / max_dimension, width / max_dimension, width)
    y = np.linspace(-height / max_dimension, height / max_dimension, height)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    if not center is None:
        rh = height / max_dimension
        rv = width / max_dimension
        center_x = float(center[0] * 2) / max_dimension - rh
        center_y = float(center[1] * 2) / max_dimension - rv
        xv -= center_y
        yv -= center_x
    return np.stack([xv, yv])


## -------------------- Functions with return a vector field ---------------------


def make_flat_zoom_vector_field(width, height, zoom_factor=1.05, center=None):
    """Make a grid of displacement vectors simulating a flat zoom.

    args:
      width
      height
      zoom_constant
      center (x,y) in pixels
    """
    grid = unit_grid(width, height, center)
    # Compute displacement
    grad = -grid * zoom_factor
    return grad


def make_perspective_zoom_vector_field(width, height, zoom_constant=0.25, zoom_factor=0.85, center=None):
    """Make a grid of displacement vectors simulating a perspective zoom."""
    grid = unit_grid(width, height, center)
    r = np.sqrt(np.square(grid[0]) + np.square(grid[1])) + zoom_constant
    # Compute displacement
    grad = -grid * zoom_factor * r
    return grad


def make_radial_flower_vector_field(width, height, zoom_constant=0.25, zoom_factor=0.85, petals=8):
    """Make a grid of displacement vectors simulating a perspective zoom."""
    grid = unit_grid(width, height)
    theta = np.arctan2(grid[0], grid[1])
    r = np.sqrt(np.square(grid[0]) + np.square(grid[1])) + zoom_constant
    # Compute displacement
    grad = -grid * zoom_factor * r * (np.sin(theta * petals) + 1.5)
    return grad


def make_y_0_vector_field(width, height):
    """Make a transformation where the x speed is proportional to the y coordinate."""
    grid = unit_grid(width, height)
    dx = grid[1]
    dy = np.zeros_like(grid[1])
    return np.stack([dx, dy])


def make_rotation_field(width, height, center, clockwise=True):
    """Make a transformation with a 1/r rotation centerated at point center.

    Args:
      width
      height
      center: (x, y) in (-1, 1) space.
    """
    grid = unit_grid(width, height)
    xv = grid[0]
    yv = grid[1]
    xv -= center[1]
    yv -= center[0]
    r = np.sqrt(np.square(xv) + np.square(yv))
    if clockwise:
        dx = -yv / r
        dy = xv / r
    else:
        dx = yv / r
        dy = -xv / r
    return np.stack([dx, dy])


## -------------------- Functions to transform an image according to a vector field transform ---------------------


def vector_field_transform(image, grad):
    """Transform an image's coordinates accoring to a vector field.
    dx, dy specify how much to displace each input point to its corresponding output point.

    Returns:
      transformed image
    """

    def mapping(output_coords):
        x = output_coords[0]
        y = output_coords[1]
        dx = grad[0, x, y]
        dy = grad[1, x, y]
        return x + dx, y + dy, output_coords[2]

    return scipy.ndimage.geometric_transform(image, mapping, output_shape=image.shape, mode='nearest')


def perspective_zoom(image, zoom_constant=0.25, zoom_factor=0.85, center=None):
    """Zoom an image using a perspective transformation with vanishing point at the center.
       Creates a false sense of depth.

       Args:
         image - an array of shape (width, height, channels)
         zoom_factor - the amount by which to zoom in.
    """
    grad = make_perspective_zoom_vector_field(image.shape[0], image.shape[1],
                                              zoom_constant=zoom_constant, zoom_factor=zoom_factor, center=center)
    return vector_field_transform(image, grad)
