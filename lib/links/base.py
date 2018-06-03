import chainer

from .layer import Convolution, Shortcut, Route, Upsample
from .yolo import YOLO

    
def forward_layer(layer, x, hs):
    if isinstance(layer, (Convolution, Upsample, YOLO)):
        x = layer(x)
    elif isinstance(layer, (Shortcut, Route)):
        x = layer(hs)
    else:
        raise Exception(layer)
    
    return x

class YOLOv3Base(chainer.Chain):
    
    def __init__(self):
        super(YOLOv3Base, self).__init__()
        
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.layers = chainer.ChainList(*[
                Convolution(3, 32, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(32, 64, 3, 2, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(64, 32, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(32, 64, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(64, 128, 3, 2, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 64, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(64, 128, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(128, 64, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(64, 128, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 10
                Shortcut(-3),
                Convolution(128, 256, 3, 2, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 20
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),                                                                                   # 30
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 128, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(128, 256, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(256, 512, 3, 2, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),                                                                                   # 40
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer), # 50
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(512, 256, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(256, 512, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 60
                Shortcut(-3),
                Convolution(512, 1024, 3, 2, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer), # 70
                Shortcut(-3),
                Convolution(1024, 512, 1, 1, 0, batch_normalize=True, activation='leaky', initialW=initializer),
                Convolution(512, 1024, 3, 1, 1, batch_normalize=True, activation='leaky', initialW=initializer),
                Shortcut(-3)
            ])
        return
    
    def __call__(self, x):
        hs = []
        
        for layer in self.layers:
            x = forward_layer(layer, x, hs)
            hs.append(x)
        
        return x, hs
        