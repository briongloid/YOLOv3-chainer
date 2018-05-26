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
from lib.visualize import vis_yolo

def main():
    parser = argparse.ArgumentParser(description='Chainer YOLOv3 Predict')
    parser.add_argument('--yolo')
    parser.add_argument('--image')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='detect')
    parser.add_argument('--thresh', type=float, default=0.8)
    args = parser.parse_args()
    
    print('yolo weight:', args.yolo)
    print('image path:', args.image)
    print('GPU:', args.gpu)
    print('out:', args.out)
    print('thresh:', args.thresh)
    print('')
    
    print('Loading Weight')
    yolov3 = YOLOv3(20)
    serializers.load_npz(args.yolo, yolov3)
    print('Loaded Weight')
    
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        yolov3.to_gpu()
    
    detector = YOLOv3Predictor(yolov3, thresh=args.thresh)
    class_names = load_list('./data/voc.names')
    
    org_image = np.array(Image.open(args.image))
    print('original size: {}'.format(org_image.shape))
    h, w, _ = org_image.shape
    ch = h//32*32
    cw = w//32*32

    top = (h-ch)//2
    left = (w-cw)//2
    bottom = top+ch
    right = left+cw

    org_image = org_image[top:bottom, left:right, :]
    print('cropped size: {}'.format(org_image.shape))
    
    image = org_image.astype(np.float32)/255.0
    image = image.transpose(2,0,1)
    batch = [image]
    batch = convert.concat_examples(batch, args.gpu)
    
    start = time.time()
    print('First Detection Start')
    with chainer.using_config('train', False), \
         chainer.no_backprop_mode():
        dets = detector(batch)[0]
    elapsed_time = time.time() - start
    print('First Detection End')
    print('elapsed time: {}s'.format(elapsed_time))
    
    start = time.time()
    print('Second Detection Start')
    with chainer.using_config('train', False), \
         chainer.no_backprop_mode():
        dets = detector(batch)[0]
    elapsed_time = time.time() - start
    print('Second Detection End')
    print('elapsed time: {}s'.format(elapsed_time))
    
    bboxes = []
    confs = []
    probs = []
    for det in dets:
        #box = cuda.to_cpu(det['box'])
        #conf = cuda.to_cpu(det['conf'])
        #prob = cuda.to_cpu(det['prob'])
        box = np.array(det['box'])
        conf = np.array(det['conf'])
        prob = np.array(det['prob'])
        prob = prob * conf
        prob[prob<args.thresh] = 0

        bboxes.append(box)
        confs.append(conf)
        probs.append(prob)
    
    for i, (bbox, conf, prob) in enumerate(zip(bboxes, confs, probs)):
        text = 'conf {:.3}'.format(float(conf))
        prob_texts = []
        for j, p in enumerate(prob):
            if p >= args.thresh:
                prob_texts.append('{} {:.3}'.format(class_names[j], float(p)))
        if len(prob_texts) > 0:
            text += ':' + ','.join(prob_texts)
        else:
            continue
        print(text)
    
    det_image = vis_yolo(org_image, 
                         bboxes, confs, probs,
                         class_names, args.thresh)
    os.makedirs(args.out, exist_ok=True)
    det_image = Image.fromarray(det_image)
    det_image.save(os.path.join(args.out, os.path.basename(args.image)))

if __name__ == '__main__':
    main()