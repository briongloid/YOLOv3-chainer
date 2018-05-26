import numpy as np

def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = np.max([x1 - len1_half, x2 - len2_half])
    right = np.max([x1 + len1_half, x2 + len2_half])

    return right - left

def box_iou(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    w = np.max([w, 0.0])
    h = np.max([h, 0.0])
    
    area_i = w*h
    area_a = a[2]*a[3]
    area_b = b[2]*b[3]
    return area_i / (area_a+area_b-area_i)