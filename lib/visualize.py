import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from lib.utils import yolo2xyminmax, detection2text

_colors = plt.get_cmap('tab20').colors
_colors = np.array(_colors) * 255
_colors = _colors.astype(np.int)
_colors = [tuple(_colors[i].tolist()) for i in range(len(_colors))]
_colors_bgr = [_colors[i][::-1] for i in range(len(_colors))]

def vis_yolo(image, bboxes, confs, probs, names, thresh, image_is_bgr=False):
    font = cv2.FONT_HERSHEY_PLAIN
    image = np.array(image)
    h, w, _ = image.shape
    size = (w, h)
    if image_is_bgr:
        colors = _colors_bgr
    else:
        colors = _colors
    
    for i, (bbox, conf, prob) in enumerate(zip(bboxes, confs, probs)):
        text = detection2text(conf, prob, names, thresh)
        if len(text) == 0:
            continue
        
        max_i = np.argmax(prob)
        color_i = max_i % len(colors)
        bbox = yolo2xyminmax(size, bbox)
        left, right, top, bottom = bbox
        left = max(left, 0)
        right = min(right, w-1)
        top = max(top, 0)
        bottom = min(bottom, h-1)
        
        cv2.rectangle(image, (left, top), (right, bottom), colors[color_i], 2)
        
        cv2.rectangle(image, (left, top-12), (left+len(text)*9, top), colors[color_i], -1)
        cv2.putText(image, text, (left, top), font, 1, (0,0,0), 1)
    
    return image