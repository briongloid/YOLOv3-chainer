import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

from lib.utils import yolo2xyminmax, detection2text

def _get_color(i, n_class, cmap, image_is_bgr):
    color = cmap(i / n_class)[1:]
    color = tuple((np.array(color)*255).astype(np.int).tolist())
    if image_is_bgr:
        color = color[::-1]
    return color

def vis_yolo(image, bboxes, confs, probs, names, thresh, cmap=plt.cm.rainbow, image_is_bgr=False):
    font = cv2.FONT_HERSHEY_PLAIN
    image = np.array(image)
    h, w, _ = image.shape
    size = (w, h)
    n_class = len(names)
    
    for i, (bbox, conf, prob) in enumerate(zip(bboxes, confs, probs)):
        text = detection2text(conf, prob, names, thresh)
        if len(text) == 0:
            continue
        
        max_i = np.argmax(prob)
        color = _get_color(max_i, n_class, cmap, image_is_bgr)
        
        bbox = yolo2xyminmax(size, bbox)
        left, right, top, bottom = bbox
        left = max(left, 0)
        right = min(right, w-1)
        top = max(top, 0)
        bottom = min(bottom, h-1)
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        cv2.rectangle(image, (left, top-12), (left+len(text)*9, top), color, -1)
        cv2.putText(image, text, (left, top), font, 1, (0,0,0), 1)
    
    return image

def vis_bbox(image, bboxes, labels, names, cmap=plt.cm.rainbow, image_is_bgr=False):
    font = cv2.FONT_HERSHEY_PLAIN
    image = np.array(image)
    h, w, _ = image.shape
    size = (w, h)
    n_class = len(names)
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        text = names[label]
        color = _get_color(label, n_class, cmap, image_is_bgr)
        
        bbox = yolo2xyminmax(size, bbox)
        left, right, top, bottom = bbox
        left = max(left, 0)
        right = min(right, w-1)
        top = max(top, 0)
        bottom = min(bottom, h-1)
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        cv2.rectangle(image, (left, top-12), (left+len(text)*9, top), color, -1)
        cv2.putText(image, text, (left, top), font, 1, (0,0,0), 1)
    
    return image
