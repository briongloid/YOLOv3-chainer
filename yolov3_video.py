import argparse
import os
import sys

import numpy as np
import cv2

import chainer
from chainer import serializers
from chainer.backends import cuda
from chainer.dataset import convert

from lib.links.yolov3 import YOLOv3
from lib.links.predict import YOLOv3Predictor
from lib.data import load_list
from lib.image import letterbox_image
from lib.visualize import vis_yolo

def main():
    parser = argparse.ArgumentParser(description='Chainer YOLOv3 Video')
    parser.add_argument('--yolo')
    parser.add_argument('--names')
    parser.add_argument('--video')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='video')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    class_names = load_list(args.names)
    
    print('Loading Weight')
    yolov3 = YOLOv3(len(class_names))
    serializers.load_npz(args.yolo, yolov3)
    print('Loaded Weight')

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        yolov3.to_gpu()

    detector = YOLOv3Predictor(yolov3, thresh=args.thresh)
    
    os.makedirs(args.out, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(args.video))
    output_path = os.path.join(args.out, '{}_yolov3{}'.format(name, ext))
    
    count = 0
    cap = cv2.VideoCapture(args.video)

    video = None
    size = None

    while cap.isOpened():

        sys.stdout.write('\rframe: {}'.format(count))
        sys.stdout.flush()
        flag, frame = cap.read()

        if video is None:
            h, w, _ = frame.shape
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            video = cv2.VideoWriter(output_path, fourcc, 20.0, size, True)

        if flag == False:
            break

        image = letterbox_image(frame[:,:,::-1], (416, 416))
        image = image.astype(np.float32)/255.0
        image = image.transpose(2,0,1)
        batch = [image]
        batch = convert.concat_examples(batch, args.gpu)

        with chainer.using_config('train', False), \
             chainer.no_backprop_mode():
            dets = detector(batch, size)[0]

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

        det_image = vis_yolo(frame, 
                             bboxes, confs, probs,
                             class_names, args.thresh, image_is_bgr=True)

        video.write(det_image)
        count += 1

    print('')
    
    cap.release()
    video.release()
    
    

if __name__ == '__main__':
    main()