import numpy as np
from chainer.dataset import concat_examples, to_device

def concat_yolo(batch, device=None):
    images = []
    bbox = []
    label = []
    for b in batch:
        images.append(b[0])
        bbox.append(to_device(device, np.array(b[1], dtype=np.float32)))
        label.append(to_device(device, np.array(b[2], dtype=np.int32)))
    images = concat_examples(images, device)
    return images, bbox, label
