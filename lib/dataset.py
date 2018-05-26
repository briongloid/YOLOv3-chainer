import numpy as np
from chainer import configuration
from chainer.dataset.dataset_mixin import DatasetMixin

from .image import random_detection
from .utils import xyminmax2yolo

class YOLOVOCDataset(DatasetMixin):
    
    def __init__(self, dataset, classifier=True,
                 crop_size=(416, 416), jitter=0, 
                 hue=0.0, sat=1.0, val=1.0):
        super(YOLOVOCDataset, self).__init__()
        self.dataset = dataset
        self.classifier = classifier
        self.crop_size = crop_size
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

        return
    
    def __len__(self):
        return len(self.dataset)
    
    def set_crop_size(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
    
    def get_example(self, i):
        
        image, bbox, label = self.dataset[i]
        bbox, label = bbox[:90], label[:90]
        # bbox convert
        bbox = bbox[:, [1,3,0,2]]
        bbox = bbox.astype(np.float32)
        _, h, w = image.shape
        size = (w, h)
        _bbox = []
        for j in range(len(bbox)):
            _bbox.append(
                xyminmax2yolo(size, bbox[j])
            )
        bbox = np.array(_bbox).astype(np.float32)
        
        # random detection
        image = image.transpose(1, 2, 0)
        image, bbox, label = random_detection(
            image, bbox, label, self.crop_size,
            self.jitter, self.hue, self.sat, self.val
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
