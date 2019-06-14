"""Functions for reading and processing optical flow."""
import cv2
import numpy as np


def vector_field(grid, fx, frame=0):
    """Makes a vector field by evaluating fx at every point in grid.

       Args:
         grid - A grid, see image_transforms.unit_grid
         fx - A vectorized function from (x, y) -> (dx, dy)
         frame - An optional frame argument which gets forwarded to fx.
    """
    dx, dy = fx(grid[0], grid[1], frame=frame)
    return np.stack([dx, dy], axis=0)


# ------------ Helper Functions ------------

def length(x, y):
    return np.sqrt(x*x + y*y)


def power(x, y):
    return np.real(np.power(x+0j, y))


# ------------ Saved Vector Fields ------------


def alien_heads(x, y):
    dx = (np.log(np.cos(power(y, y)) * np.cos(y)) + np.sin(length(x, y)))
    dy = x * length(x, y)
    return dx, dy


def zooming_spiral(x, y, frame=0):
    r = length(x, y)
    theta = np.arctan2(y, x)
    v = np.stack([y, -x], axis=0) / r
    t = np.sqrt(r * 10.) + theta + frame * .02
    v1 = v * np.sin(t)
    v2 = v1 * length(v1[0], v1[1]) * 10.
    v3 = v2 + np.array([x, y]) * .2
    return v3[0], v3[1]


named = {
    'alien_heads': alien_heads,
    'zooming_spiral': zooming_spiral,
}
