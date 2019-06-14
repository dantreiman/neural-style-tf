"""Fast implementations of common image ops (rotate, scale, remap).

I had originaly implemented these using skimage.transform and scikit.ndimage.geometric_transform,
however OpenCV's implementations are orders of magnitude faster.

For example, to remap a 6000x4000 image using bicubic interpolation:

scipy.ndimage.geometric_transform: 178.1593s
cv2.remap: 0.5271s
cv2.remap + convertMaps: 0.4053s
"""
import cv2
import numpy as np


def grad_to_cv_map(grad):
    """Make openCV map from a displacement map.

       grad: (2, width, height) Gradient computed at each point with respect to x and y.
    """
    x = np.arange(0, grad.shape[1])
    y = np.arange(0, grad.shape[2])
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    mx = xv + grad[0]
    my = yv + grad[1]
    transform = np.stack((mx, my), axis=2).astype(np.float32)
    tform_cv_x = transform[:, :, 1]
    tform_cv_y = transform[:, :, 0]
    cvmap1, cvmap2 = cv2.convertMaps(tform_cv_x, tform_cv_y, cv2.CV_16SC2)
    return (cvmap1, cvmap2)


def rescale(image, scale_factor):
    """Rescale image using scale factor."""
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def resize(image, size):
    """Resize image to desired size."""
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def rotate(image, angle):
    """Rotate image by specified angle"""
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), cv2.INTER_CUBIC)


def remap(image, transform):
    """Remap an image using the specified transform, as a pair of opencv int matrices.
    Generate transform using grad_to_cv_map.

    Returns:
      transformed image
    """
    return cv2.remap(image, transform[0], transform[1], interpolation=cv2.INTER_LINEAR)


def convert_colors(content_img, stylized_img, color_space='yuv'):
    """Convert one image's colors to the color space of the other."""
    if color_space == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_space == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_space == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_space == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    return dst
