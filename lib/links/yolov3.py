import chainer

from .base import forward_layer, YOLOv3Base
from .layer import Convolution, Shortcut, Route, Upsample
from .yolo import YOLO

class YOLOv3(chainer.Chain):
    
    def __init__(self, n_class, base=None, ignore_thresh=0.7):
        super(YOLOv3, self).__init__()
        
        self.n_class = n_class
        anchors = [
            [10,13], [16,30], [33,23],
            [30,61], [62,45], [59,119],
            [116,90], [156,198], [373,326]
        ]
        
        yolo_channels = 3 * (5 + n_class)
        initializer = chainer.initializers.HeNormal()
        
        with self.init_scope():
            if base is None:
                self.base = YOLOv3Base()
            else:
                self.base = base
            self.layers = chainer.ChainList(*[
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 80
                Convolution(1024, yolo_channels, 1, 1, 0, initialW=initializer),
                YOLO(n_class, anchors, [6,7,8], ignore_thresh=ignore_thresh),
                Route(-4),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Upsample(2),
                Route([-1, 61]),
                Convolution(768, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 90
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, yolo_channels, 1, 1, 0, initialW=initializer),
                YOLO(n_class, anchors, [3,4,5], ignore_thresh=ignore_thresh),
                Route(-4),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Upsample(2),
                Route([-1, 36]),
                Convolution(384, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 100
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, yolo_channels, 1, 1, 0, initialW=initializer),
                YOLO(n_class, anchors, [0,1,2], ignore_thresh=ignore_thresh)                                     # 106
            ])
        return
    
    def __call__(self, x):
        x, hs = self.base(x)
        ys = []
        for layer in self.layers:
            x = forward_layer(layer, x, hs)
            hs.append(x)
            
            if isinstance(layer, YOLO):
                ys.append({
                    'y': x,
                    'layer': layer
                })
        
        return ys
            
            
        