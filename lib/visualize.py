import numpy as np
from PIL import Image
import cv2

from lib.utils import yolo2xyminmax, detection2text

def vis_yolo(image, bboxes, confs, probs, names, thresh):
    font = cv2.FONT_HERSHEY_PLAIN
    image = np.array(image)
    h, w, _ = image.shape
    size = (w, h)
    
    for i, (bbox, conf, prob) in enumerate(zip(bboxes, confs, probs)):
        text = detection2text(bbox, conf, prob, names, thresh)
        if len(text) == 0:
            continue
        
        bbox = yolo2xyminmax(size, bbox)
        left, right, top, bottom = bbox
        left = max(left, 0)
        right = min(right, w-1)
        top = max(top, 0)
        bottom = min(bottom, h-1)
        
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        
        cv2.rectangle(image, (left, top-12), (left+len(text)*9, top), (255, 255, 255), -1)
        cv2.putText(image, text, (left, top), font, 1, (0,0,0), 1)
    
    return image