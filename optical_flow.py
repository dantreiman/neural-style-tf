"""Functions for reading and processing optical flow."""
import cv2
import numpy as np
import struct


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
