import argparse
import os
import random

import numpy as np
import chainer
import chainercv
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
from chainercv.datasets import VOCBboxDataset

from lib.links.darknet53 import Darknet53
from lib.datasets.yolo_voc_dataset import YOLOVOCDataset
from lib.extensions.darknet_shift import DarknetShift
from lib.extensions.crop_size_updater import CropSizeUpdater

def main():
    parser = argparse.ArgumentParser(description='Chainer Darknet53 Train')
    parser.add_argument('--batchsize', '-b', type=int, default=8)
    parser.add_argument('--iteration', '-i', type=int, default=100000)
    parser.add_argument('--gpus', '-g', type=int, nargs='*', default=[])
    parser.add_argument('--out', '-o', default='darknet53-voc-result')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--snapshot_interval', type=int, default=100)
    parser.add_argument('--validation_size', type=int, default=2048)
    args = parser.parse_args()

    print('GPUs: {}'.format(args.gpus))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))
    print('')

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    darknet53 = Darknet53(20)
    model = L.Classifier(darknet53)
    device = -1
    if len(args.gpus) > 0:
        device = args.gpus[0]
        cuda.cupy.random.seed(args.seed)
        cuda.get_device_from_id(args.gpus[0]).use()
    if len(args.gpus) == 1:
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(
        chainer.optimizer_hooks.WeightDecay(0.0005), 'hook_decay')

    train = VOCBboxDataset(split='train')
    test = VOCBboxDataset(split='val')
    train = YOLOVOCDataset(train, classifier=True, jitter=0.2,
                        hue=0.1, sat=.75, val=.75)
    test = YOLOVOCDataset(test, classifier=True, crop_size=(256, 256))
    test = test[np.random.permutation(np.arange(len(test)))[:min(args.validation_size, len(test))]]

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    if len(args.gpus) <= 1:
        updater = training.StandardUpdater(
            train_iter, optimizer, device=device)
    else:
        devices = {'main': args.gpus[0]}
        for gpu in args.gpus[1:]:
            devices['gpu{}'.format(gpu)] = gpu
        updater = training.ParallelUpdater(
            train_iter, optimizer, devices=devices)
    
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)
    
    display_interval = (args.display_interval, 'iteration')
    snapshot_interval = (args.snapshot_interval, 'iteration')
    
    trainer.extend(extensions.Evaluator(test_iter, model, device=device), 
                   trigger=display_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'iteration', display_interval, file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'iteration', display_interval, file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
                  trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(extensions.snapshot_object(
        darknet53, 'darknet53_snapshot.npz'), 
        trigger=training.triggers.MinValueTrigger(
            'validation/main/loss', snapshot_interval))
    trainer.extend(extensions.snapshot_object(
        darknet53, 'darknet53_final.npz'), 
        trigger=snapshot_interval)
    
    trainer.extend(DarknetShift(optimizer, 'poly', args.iteration))
    
    trainer.extend(CropSizeUpdater(train, [(4+i)*32 for i in range(0,11)]))
    
    trainer.run()

if __name__ == '__main__':
    main()