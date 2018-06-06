import argparse
import os
import time

import numpy as np
from PIL import Image

import chainer
from chainer import serializers
from chainer.backends import cuda
from chainer.dataset import convert

from lib.links.yolov3 import YOLOv3
from lib.links.loss import YOLOv3Loss
from lib.links.predict import YOLOv3Predictor
from lib.data import load_list
from lib.image import letterbox_image
from lib.utils import detection2text
from lib.visualize import vis_yolo

def main():
    parser = argparse.ArgumentParser(description='Chainer YOLOv3 VOC Predict')
    parser.add_argument('--yolo')
    parser.add_argument('--names')
    parser.add_argument('--image')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='detect')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    print('yolo weight:', args.yolo)
    print('image path:', args.image)
    print('GPU:', args.gpu)
    print('out:', args.out)
    print('thresh:', args.thresh)
    print('')
    
    class_names = load_list(args.names)
    print('Loading Weight')
    yolov3 = YOLOv3(len(class_names))
    serializers.load_npz(args.yolo, yolov3)
    print('Loaded Weight')
    
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        yolov3.to_gpu()
    
    detector = YOLOv3Predictor(yolov3, thresh=args.thresh)
    
    org_image = np.array(Image.open(args.image))
    org_size = org_image.shape[1::-1]
    image = letterbox_image(org_image, (416, 416))
    image = image.astype(np.float32)/255.0
    image = image.transpose(2,0,1)
    batch = [image]
    batch = convert.concat_examples(batch, args.gpu)
    
    start = time.time()
    print('First Detection Start')
    with chainer.using_config('train', False), \
         chainer.no_backprop_mode():
        dets = detector(batch, org_size)[0]
    elapsed_time = time.time() - start
    print('First Detection End')
    print('elapsed time: {}s'.format(elapsed_time))
    
    start = time.time()
    print('Second Detection Start')
    with chainer.using_config('train', False), \
         chainer.no_backprop_mode():
        dets = detector(batch, org_size)[0]
    elapsed_time = time.time() - start
    print('Second Detection End')
    print('elapsed time: {}s'.format(elapsed_time))
    
    bboxes = []
    confs = []
    probs = []
    for det in dets:
        bbox = det['box']
        conf = det['conf']
        prob = det['prob']

        bboxes.append(bbox)
        confs.append(conf)
        probs.append(prob)
        
        text = detection2text(conf, prob, class_names, args.thresh)
        if len(text) > 0:
            print(text)
    
    det_image = vis_yolo(org_image, 
                         bboxes, confs, probs,
                         class_names, args.thresh)
    os.makedirs(args.out, exist_ok=True)
    det_image = Image.fromarray(det_image)
    det_image.save(os.path.join(args.out, os.path.basename(args.image)))

if __name__ == '__main__':
    main()