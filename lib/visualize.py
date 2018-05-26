import numpy as np
from PIL import Image
import cv2

from lib.utils import yolo2xyminmax

def vis_yolo(image, bboxes, confs, probs, names, thresh):
    font = cv2.FONT_HERSHEY_PLAIN
    image = np.array(image)
    h, w, _ = image.shape
    size = (w, h)
    
    for i, (bbox, conf, prob) in enumerate(zip(bboxes, confs, probs)):
        text = 'conf {:.3}'.format(float(conf))
        prob_texts = []
        for j, p in enumerate(prob):
            if p >= thresh:
                prob_texts.append('{} {:.3}'.format(names[j], float(p)))
        if len(prob_texts) > 0:
            text += ':' + ','.join(prob_texts)
        else:
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