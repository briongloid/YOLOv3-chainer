import numpy as np
from PIL import Image
import cv2

from .utils import rand_scale

def random_boxes(box, label):
    index = np.arange(len(label))
    index = np.random.permutation(index)
    return box[index], label[index]

def correct_boxes(box, crop_size, new_size, start_point, flip):
    box = np.array(box)
    x, y, w, h = box.transpose(1, 0)
    crop_size = np.array(crop_size).astype(np.float32)
    sx, sy = new_size/crop_size
    dx, dy = start_point/crop_size
    
    left = (x - w/2) * sx + dx
    right = (x + w/2) * sx + dx
    top = (y - h/2) * sy + dy
    bottom = (y + h/2) * sy + dy
    
    if flip:
        left, right = 1. - right, 1. - left
    
    left, right, top, bottom = np.clip([left, right, top, bottom], 0, 1)
    
    box[:, 0] = (left+right)/2
    box[:, 1] = (top+bottom)/2
    box[:, 2] = (right-left)
    box[:, 3] = (bottom-top)
    
    box[:, 2:4] = np.clip(box[:, 2:4], 0, 1)
    return box

def reselect_boxes(box, label, crop_size, new_size, start_point, flip):
    box, label = random_boxes(box, label)
    box = correct_boxes(box, crop_size, new_size, start_point, flip)
    
    select_index = np.logical_and(box[:, 2]>=0.005, box[:, 3]>=0.005)
    return box[select_index], label[select_index]

def get_pad_slice(start, crop_length, org_length):
    if crop_length>=org_length:
        return (start, crop_length-org_length-start), slice(0, crop_length)
    else:
        return (0, 0), slice(-start, -start+crop_length)

def resize_crop_image(image, crop_size, new_size, start_point):
    image = cv2.resize(image, new_size)
    pw, cw = get_pad_slice(start_point[0], crop_size[0], new_size[0])
    py, cy = get_pad_slice(start_point[1], crop_size[1], new_size[1])
    image = np.pad(image, [py, pw, (0,0)],'constant', constant_values=127)
    image = image[cy, cw]
    return image

def random_hsv_image(rgb_image, hue=0, sat=1, val=1):
    
    if 0==hue and 1==sat and 1==val:
        return rgb_image
    
    dtype = rgb_image.dtype
    rgb_image = rgb_image.astype(np.uint8)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float)
    
    if 0 != hue:
        dhue = np.random.uniform(-hue, hue)
        hsv_image[:,:,0] += 180*dhue
        hsv_image[:,:,0] %= 180
    
    if 1 != sat:
        dsat = rand_scale(sat)
        hsv_image[:,:,1] *= dsat
    
    if 1 != val:
        dval = rand_scale(val)
        hsv_image[:,:,2] *= dval

    hsv_image = np.clip(hsv_image, 0, 255)
    hsv_image = hsv_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    rgb_image = rgb_image.astype(dtype)
    return rgb_image

def random_detection(image, bbox, label, crop_size, jitter, hue, sat, val):
    image = np.array(image)
    w, h = crop_size
    
    dw, dh = jitter*image.shape[1], jitter*image.shape[0]
    new_ar = (image.shape[1] + np.random.uniform(-dw, dw)) / (image.shape[0] + np.random.uniform(-dh, dh))
    scale = np.random.uniform(0.5, 2)
    
    if new_ar < 1:
        nh = scale * h
        nw = nh * new_ar
    else:
        nw = scale * w
        nh = nw / new_ar
    
    dx = np.random.uniform(0, w - nw)
    dy = np.random.uniform(0, h - nh)
    
    new_size = (round(nw), round(nh))
    start_point = (round(dx), round(dy))
    image = resize_crop_image(image, crop_size, new_size, start_point)
    image = random_hsv_image(image, hue, sat, val)
    
    flip = np.random.uniform()>0.5
    if flip:
        image = image[:,::-1]
    
    bbox, label = reselect_boxes(bbox, label, crop_size, new_size, start_point, flip)
    
    return image, bbox, label