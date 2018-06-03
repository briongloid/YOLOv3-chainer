import os

import numpy as np
from PIL import Image
import cv2

import chainer
from chainer.backends import cuda
from chainer.dataset import convert
from chainer.training import extension
from chainer.training import trigger as trigger_module

from ..image import letterbox_image
from ..utils import yolo2xyminmax
from ..visualize import vis_yolo

class YOLODetection(extension.Extension):
    
    def __init__(self, detector, image_paths, names, size, thresh=0.6,
                 trigger=(1, 'epoch'), device=-1):
        self._detector = detector
        self._image_paths = image_paths
        self._names = names
        self._size = size
        self._thresh = thresh
        self._trigger = trigger_module.get_trigger(trigger)
        self._device = device
    
    def __call__(self, trainer):
        
        if self._trigger(trainer):
            for i in range(len(self._image_paths)):
                org_image = np.array(Image.open(self._image_paths[i]))
                org_size = org_image.shape[1::-1]
                image = letterbox_image(org_image, self._size)
                image = image.astype(np.float32)/255.0
                image = image.transpose(2,0,1)
                batch = [image]
                batch = convert.concat_examples(batch, self._device)
                
                with chainer.using_config('train', False), \
                     chainer.no_backprop_mode():
                    dets = self._detector(batch, org_size)[0]
                bboxes = []
                confs = []
                probs = []
                for det in dets:
                    box = np.array(det['box'])
                    conf = np.array(det['conf'])
                    prob = np.array(det['prob'])
                    prob = prob * conf
                    prob[prob<self._thresh] = 0
                    
                    bboxes.append(box)
                    confs.append(conf)
                    probs.append(prob)
                
                det_image = vis_yolo(org_image, 
                                     bboxes, confs, probs,
                                     self._names, self._thresh)
                det_image = Image.fromarray(det_image)
                out_dir = os.path.join(trainer.out, 'test_detection')
                os.makedirs(out_dir, exist_ok=True)
                
                name, fextension = os.path.splitext(
                    os.path.basename(self._image_paths[i]))
                save_name = '{}_{:0>8}{}'.format(
                    name, trainer.updater.iteration, fextension)
                save_path = os.path.join(out_dir, save_name)
                det_image.save(save_path)
                