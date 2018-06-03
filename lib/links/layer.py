import numpy as np
import cupy

import chainer
import chainercv
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda

class Convolution(chainer.Chain):
    
    def __init__(self, in_channels, out_channels,
                 ksize=None, stride=1, pad=0, 
                 batch_normalize=False, activation='linear', initialW=None):
        super(Convolution, self).__init__()
        
        self.batch_normalize = batch_normalize
        
        if activation not in ['linear', 'leaky']:
            raise ValueError()
        
        if 'linear' == activation:
            self.activation = F.identity
        elif 'leaky' == activation:
            self.activation = lambda x: F.leaky_relu(x, slope=0.1)
        
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, 
                                        ksize, stride, pad, 
                                        nobias=True,
                                        initialW=initialW)
            if batch_normalize:
                self.bn = L.BatchNormalization(out_channels)
            
            self.b = L.Bias(shape=(out_channels,))
        
    def __call__(self, x):
        h = self.conv(x)
        if self.batch_normalize:
            h = self.bn(h)
        h = self.b(h)
        h = self.activation(h)
        return h

class Shortcut(chainer.Chain):
    
    def __init__(self, layer_index):
        super(Shortcut, self).__init__()
        self.layer_index = layer_index
    
    def __call__(self, hs):
        return hs[-1] + hs[self.layer_index]

class Route(chainer.Chain):
    
    def __init__(self, layer_index):
        super(Route, self).__init__()
        if isinstance(layer_index, int):
            layer_index = [layer_index]
        self.layer_index = layer_index
    
    def __call__(self, hs):
        #for i in self.layer_index:
        #    print(hs[i].shape)
        return F.concat([hs[i] for i in self.layer_index], axis=1)

class Upsample(chainer.Chain):
    def __init__(self, stride):
        super(Upsample, self).__init__()
        self.stride = stride
        
    def __call__(self, x):
        return F.unpooling_2d(x, self.stride, 
                              outsize=np.array(x.shape[2:])*self.stride)

