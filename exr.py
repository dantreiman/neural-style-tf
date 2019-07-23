import Imath
import numpy as np
import OpenEXR


def load_depth_file(path):
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_file = OpenEXR.InputFile(path)
    header = depth_file.header()
    data_window = header['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    r = np.fromstring(depth_file.channel('R', FLOAT), dtype=np.float32)
    g = np.fromstring(depth_file.channel('G', FLOAT), dtype=np.float32)
    b = np.fromstring(depth_file.channel('B', FLOAT), dtype=np.float32)
    a = np.fromstring(depth_file.channel('A', FLOAT), dtype=np.float32)
    depth_file.close()
    r.shape = g.shape = b.shape = a.shape = (size[1], size[0])
    return np.dstack([r, g, b, a])

