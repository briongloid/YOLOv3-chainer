import numpy as np
from PIL import Image

import chainer
from chainer.dataset.dataset_mixin import DatasetMixin

from ..data import load_list
from ..image import random_detection, test_detection

def _to_label_path(image_path):
    label_path = image_path
    label_path = label_path.replace('images', 'labels')
    label_path = label_path.replace('JPEGImages', 'labels')
    label_path = label_path.replace('raw', 'labels')

    label_path = label_path.replace('.jpg', '.txt')
    label_path = label_path.replace('.png', '.txt')
    label_path = label_path.replace('.JPG', '.txt')
    label_path = label_path.replace('.JPEG', '.txt')
    return label_path

def _load_label(label_path):
    labels = load_list(label_path)
    
    bboxes = []
    class_ids = []
    
    for label in labels:
        class_id, x, y, w, h = label.split()
        bboxes.append([float(e) for e in [x, y, w, h]])
        class_ids.append(int(class_id))
    
    return bboxes, class_ids
    
class YOLODataset(DatasetMixin):
    
    def __init__(self, path, train, classifier=False,
                 crop_size=(416, 416), jitter=0,
                 hue=0.0, sat=1.0, val=1.0):
        super(YOLODataset, self).__init__()
        self.image_paths = load_list(path)
        self.train = train
        self.classifier = classifier
        self.crop_size = crop_size
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

        
    def __len__(self):
        return len(self.image_paths)
    
    def set_crop_size(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

    def get_example(self, i):
        image_path = self.image_paths[i]
        label_path = _to_label_path(image_path)
        
        image = Image.open(image_path)
        bbox, label = _load_label(label_path)
        bbox = np.array(bbox, dtype=np.float32)
        label = np.array(label, dtype=np.int32)
        
        if self.train:
            image, bbox, label = random_detection(
                image, bbox, label, self.crop_size,
                self.jitter, self.hue, self.sat, self.val
            )
        else:
            image, bbox, label = test_detection(
                image, bbox, label, self.crop_size,
            )
        image = image.transpose(2, 0, 1)
        
        image = image/255.0
        
        if self.classifier and 0 == len(label):
            return self.get_example(i)
        
        if self.classifier:
            bbox = np.array(bbox)
            i = np.argmax(bbox[:, 2] * bbox[:, 3])
            label = label[i]
            return image, label
        else:
            return image, bbox, label