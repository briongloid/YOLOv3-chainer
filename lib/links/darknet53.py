import chainer
import chainer.functions as F

from .layer import Convolution
from .base import YOLOv3Base

class Darknet53(chainer.Chain):
    
    def __init__(self, n_class ,base=None):
        super(Darknet53, self).__init__()
        self.n_class = n_class
        
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            if base is None:
                self.base = YOLOv3Base()
            else:
                self.base = base
            self.conv = Convolution(None, n_class, 1, 1, 0, initialW=initializer)
        
        return
    
    def __call__(self, x):
        x, _ = self.base(x)
        x = F.average_pooling_2d(x, x.shape[2:4])
        x = self.conv(x)
        x = F.reshape(x, x.shape[0:2])
        return x
