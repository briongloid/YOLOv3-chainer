import argparse
import os
import sys
import time

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
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='camera_yolov3')
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
    
    os.makedirs('camera', exist_ok=True)
    output_path = os.path.join('camera', '{}.mp4'.format(args.out))
    
    count = 0
    cap = cv2.VideoCapture(args.camera)
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    start = time.time()
    
    while cap.isOpened():
        
        sys.stdout.write('\rframe: {}, elapsed time: {:.3f}s'.format(count, 
                                                                     time.time() - start))
        sys.stdout.flush()
        
        flag, frame = cap.read()

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
        
        cv2.imshow('camera capture', det_image)

        k = cv2.waitKey(1) 
        if k == 27:
            break
        
        count += 1
    
    print('')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
