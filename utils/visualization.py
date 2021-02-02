import numpy as np
import cv2


def vis_flow(flow, clip=10.0):
    flow = np.clip(flow+clip, 0, 2.0*clip) / (2.0*clip)
    flow = (flow*255).astype(np.uint8)
    flow = flow.transpose(1,2,3,0)
    flow = flow[..., [2,1,0]]
    return flow


def bitget(bitset, pos):
    offset = pos
    return bitset >> offset & 1


def bitshift(bitset, offset):
    if offset > 0:
        return bitset << offset
    else:
        return bitset >> (-offset)


def bitor(bitset_l, bitset_r):
    return bitset_l | bitset_r


def bitwise_get(bitset, pos):
    return np.bitwise_and(np.right_shift(bitset, pos), 1)


def label2color(im):
    '''
        input:
            im: numpy array of integer type
        output:
            color_map: numpy array with 3 channels
    '''
    inds = im.copy()
    r = np.zeros(im.shape, dtype=np.uint8)
    g = np.zeros(im.shape, dtype=np.uint8)
    b = np.zeros(im.shape, dtype=np.uint8)
    for i in range(8):
        np.bitwise_or(r, np.left_shift(bitwise_get(inds, 0), 7 - i), r)
        np.bitwise_or(g, np.left_shift(bitwise_get(inds, 1), 7 - i), g)
        np.bitwise_or(b, np.left_shift(bitwise_get(inds, 2), 7 - i), b)
        np.right_shift(inds, 3, inds)
    color_map = np.stack([b, g, r], axis=-1)
    return color_map

def GenColorMap(num_classes):
    color_map = []
    for i in range(num_classes):
        id = i
        r = 0
        g = 0
        b = 0
        for j in range(8):
            r = bitor(r, bitshift(bitget(id, 0), 7 - j))
            g = bitor(g, bitshift(bitget(id, 1), 7 - j))
            b = bitor(b, bitshift(bitget(id, 2), 7 - j))
            id = bitshift(id, -3)
        color_map.append((b, g, r))
    class_2_color = color_map
    color_2_class = dict([(item[1], item[0]) for item in enumerate(color_map)])
    return class_2_color, color_2_class
