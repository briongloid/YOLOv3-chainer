import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable
from chainer import reporter

class YOLOv3Loss(chainer.Chain):
    
    def __init__(self, predictor):
        super(YOLOv3Loss, self).__init__()
        with self.init_scope():
            self.predictor = predictor
    
    def __call__(self, x, tbox, tlabel):
        input_size = x.shape[2:4][::-1]
        ys = self.predictor(x)
        losses = []
        for y in ys:
            losses.append(y['layer'].get_loss(y['y'], tbox, tlabel, input_size))
        
        loss = sum(losses)
        reporter.report({'loss': loss}, self)
        return loss
