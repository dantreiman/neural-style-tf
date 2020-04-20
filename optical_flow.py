"""Functions for reading and processing optical flow."""
import cv2
import numpy as np
import struct


def read_flow_file(path):
    if path.endswith('.npy'):
        return np.load(path)
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
    try:
        with open(path, 'r') as file:
            header_line = file.readline()
            header = list(map(int, header_line.split(' ')))
            w = header[0]
            h = header[1]
            vals = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                line = file.readline().rstrip().split(' ')
                vals[i] = list(map(lambda s: np.uint8(np.float32(s)), line))
        return vals
    except Exception as ex:
        print('read_weights_file(%s) failed with exception:' % path)
        print(ex)
        return None


def resize_flow(flow, width, height):
    """Resizes optical flow to target size"""
    flow_img = np.transpose(flow, (1, 2, 0))
    scaled = cv2.resize(flow_img, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.transpose(scaled, (2, 0, 1))


def blur_fill_flow(flow, mask, steps=100):
    """Inpaint low-confidence regions of the flow file by averaging nearby regions."""
    flow_image = np.transpose(flow, [1,2,0])
    result = flow_image.copy()
    for _ in range(steps):
        result = cv2.blur(result, (101,101))
        np.putmask(result[:,:,0], mask, flow_image[:,:,0])
        np.putmask(result[:,:,1], mask, flow_image[:,:,1])
    return np.transpose(result, [2,0,1])


def _flow_inpaint(flow1d, mask, inpaint_radius=5):
    fmin,fmax = (flow1d.min(), flow1d.max())
    flow_scaled = (flow1d - fmin) * 255. / (fmax - fmin)
    flow_u8 = flow_scaled.astype(np.uint8)
    result_u8 = cv2.inpaint(flow_u8, mask, inpaint_radius, cv2.INPAINT_TELEA)
    result_1d = result_u8.astype(np.float32) * (fmax - fmin) / 255. + fmin
    return result_1d


def blur_inpaint(flow, mask):
    """Inpaint low-confidence regions of the flow file by using Alexandru Telea's fast matching method."""
    inverted_mask = 255 - mask

    # All low-confidence regions have pixel value 255 - if the low confidence regions are too small don't bother.
    if np.sum(inverted_mask) < 10000:
        return flow

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_mask = cv2.dilate(inverted_mask, dilation_kernel, iterations=2)

    flow_image = flow * (np.expand_dims(255 - dilated_mask, 0).astype(np.float32) / 255.0)
    flow_x = flow_image[0]
    flow_y = flow_image[1]

    inpainted_x = _flow_inpaint(flow_x, dilated_mask)
    inpainted_y = _flow_inpaint(flow_y, dilated_mask)

    inpainted_x = cv2.blur(inpainted_x, (51, 51))
    inpainted_y = cv2.blur(inpainted_y, (51, 51))

    result = flow.copy()
    np.putmask(result[0], dilated_mask, inpainted_x)
    np.putmask(result[1], dilated_mask, inpainted_y)
    return result


def warp_image(src, flow, interpolation=cv2.INTER_LANCZOS4):
    """Warp an image using optical flow

    Args:
        src (np.array) The source image.
        flow (np.array) The flow.
        interpolation The interpolation method, recommended values are cv2.INTER_CUBIC, cv2.INTER_AREA or cv2.INTER_LANCZOS4
    """
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1, y, :] = float(y) + flow[1, y, :]
    for x in range(w):
        flow_map[0, :, x] = float(x) + flow[0, :, x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1],
        interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
    return dst
