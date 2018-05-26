import numpy as np

import chainer
import chainer.functions as F
from chainer.backends import cuda

from ..box import box_iou

import cython
import pyximport; pyximport.install()
from ._yolo_detection import _detection
from ._yolo_loss import _t_and_scale

def _overlap(x1, len1, x2, len2):
    xp = cuda.get_array_module(x1)
    len1_half = len1/2
    len2_half = len2/2

    left = xp.maximum(x1 - len1_half, x2 - len2_half)
    right = xp.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

def _to_box(x, y, w, h, anchors, input_size):
    xp = cuda.get_array_module(x)
    w, h = xp.asarray(w), xp.asarray(h)
    shape = x.shape
    grid_h, grid_w = shape[-2:]
    anchors = xp.array(anchors, dtype=xp.float32)

    x_shift = xp.broadcast_to(xp.arange(grid_w, dtype=xp.float32), shape)
    y_shift = xp.broadcast_to(xp.arange(grid_h, dtype=xp.float32)[:,None], shape)
    w_anchor = xp.broadcast_to(anchors[:,0][:,None,None,None], shape)
    h_anchor = xp.broadcast_to(anchors[:,1][:,None,None,None], shape)

    box_x = (x + x_shift) / grid_w
    box_y = (y + y_shift) / grid_h
    box_w = xp.exp(w) * w_anchor / input_size[0]
    box_h = xp.exp(h) * h_anchor / input_size[1]
    return box_x, box_y, box_w, box_h

def _best_iou_t(x, y, w, h, anchors, tbox, input_size):
    xp = cuda.get_array_module(x)
    batchsize = x.shape[0]
    
    best_ious = []
    best_ts = []
    
    box_x, box_y, box_w, box_h = _to_box(x, y, w, h, anchors, input_size)

    for b in range(batchsize):
        if len(tbox[b]) == 0:
            best_ious.append(xp.zeros(x.shape[1:]).astype(xp.float32))
            best_ts.append(xp.zeros(x.shape[1:]).astype(xp.int32))
            continue
        ious = _box_iou((box_x[b], box_y[b], box_w[b], box_h[b]), tbox[b].transpose(1,0))
        best_ious.append(xp.max(ious, axis=0))
        best_ts.append(xp.argmax(ious, axis=0))

    best_ious = xp.array(best_ious, dtype=xp.float32)
    best_ts = xp.array(best_ts, dtype=xp.int32)
    return best_ious, best_ts

def _box_iou(a, b):
    """
    a: (x, y, w, h)
        x.shape: (n_box, 1, grid_h, grid_w)
    b: (x, y, w, h)
        x.shape: (n_truth,)
    """
    xp = cuda.get_array_module(a[0])
    a = [xp.broadcast_to(e[None], b[0].shape + e.shape) for e in a]
    b = [xp.broadcast_to(e[:,None,None,None,None], e.shape + a[0].shape[1:5]) for e in b]
    
    w = _overlap(a[0], a[2], b[0], b[2])
    h = _overlap(a[1], a[3], b[1], b[3])
    zeros = xp.zeros(w.shape).astype(w.dtype)
    w = xp.maximum(w, zeros)
    h = xp.maximum(h, zeros)
    
    area_i = w*h
    area_a = a[2]*a[3]
    area_b = b[2]*b[3]
    return area_i / (area_a+area_b-area_i)

class YOLO(chainer.Chain):
    
    def __init__(self, n_class, anchors, mask, ignore_thresh=0.7):
        super(YOLO, self).__init__()
        self.n_class = n_class
        self.anchors = anchors
        self.mask = mask
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = 1.0
        
    def __call__(self, x):
        x, y, w, h, conf, prob = self._split(x)
        
        x = F.sigmoid(x)
        y = F.sigmoid(y)
        conf = F.sigmoid(conf)
        prob = F.sigmoid(prob)
        x = self._concat(x, y, w, h, conf, prob)
        return x
    
    def _split(self, x):
        n_box = len(self.mask)
        n_class = self.n_class
        batchsize, _, grid_h, grid_w = x.shape
        
        x = F.reshape(x, (batchsize, n_box, 5+n_class, grid_h, grid_w))
        x, y, w, h, conf, prob = F.split_axis(x, (1, 2, 3, 4, 5), axis=2)
        return x, y, w, h, conf, prob
    
    def _concat(self, x, y, w, h, conf, prob):
        x = F.concat([x, y, w, h, conf, prob], axis=2)
        batchsize, n_box, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (batchsize, n_box*(5+self.n_class), grid_h, grid_w))
        return x

    def get_t_and_scale(self, x, y, w, h, conf, prob, tbox, tlabel, input_size):
        xp = cuda.get_array_module(x)
        batchsize, n_box, _, grid_h, grid_w = x.shape
        tx = xp.array(x.data)
        ty = xp.array(y.data)
        tw = xp.array(w.data)
        th = xp.array(h.data)
        tconf = xp.array(conf.data)
        tprob = xp.array(prob.data)
        
        tx = cuda.to_cpu(tx)
        ty = cuda.to_cpu(ty)
        tw = cuda.to_cpu(tw)
        th = cuda.to_cpu(th)
        tconf = cuda.to_cpu(tconf)
        tprob = cuda.to_cpu(tprob)
        
        box_learning_scale = np.tile(0, x.shape).astype(np.float32)
        anchors = np.array(self.anchors, dtype=np.int32)
        input_size = np.array(input_size, dtype=np.int32)
        abs_anchors = (anchors / input_size).astype(np.float32)
        
        for b in range(len(tbox)):
            tbox[b] = cuda.to_cpu(tbox[b])
            tlabel[b] = cuda.to_cpu(tlabel[b])
        
        tx, ty, tw, th, tconf, tprob, box_learning_scale =_t_and_scale(
            cuda.to_cpu(x.data), cuda.to_cpu(y.data), 
            cuda.to_cpu(w.data), cuda.to_cpu(h.data),
            tx, ty, tw, th, tconf, tprob,
            box_learning_scale,
            batchsize, n_box, 
            grid_h, grid_w,
            input_size, anchors, abs_anchors,
            np.array(self.mask, dtype=np.int32), self.n_class, self.ignore_thresh,
            tbox, tlabel)
        
        tx = xp.asarray(tx)
        ty = xp.asarray(ty)
        tw = xp.asarray(tw)
        th = xp.asarray(th)
        tconf = xp.asarray(tconf)
        tprob = xp.asarray(tprob)
        box_learning_scale = xp.asarray(box_learning_scale)
        
        return tx, ty, tw, th, tconf, tprob, box_learning_scale
        
    
    def _get_t_and_scale(self, x, y, w, h, conf, prob, tbox, tlabel, input_size):
        xp = cuda.get_array_module(x)
        batchsize, n_box, _, grid_h, grid_w = x.shape
        tx = xp.array(x.data)
        ty = xp.array(y.data)
        tw = xp.array(w.data)
        th = xp.array(h.data)
        tconf = xp.array(conf.data)
        tprob = xp.array(prob.data)
        
        box_learning_scale = xp.tile(0, x.shape).astype(xp.float32)
        
        best_ious, best_ts = _best_iou_t(x.data, y.data, w.data, h.data, 
                                         xp.array(self.anchors)[self.mask], 
                                         tbox, input_size)
        abs_anchors = xp.array(self.anchors) / xp.array(input_size)
        
        ignore_index = (best_ious <= self.ignore_thresh)
        tconf[ignore_index] = 0
        del ignore_index
        del best_ious, best_ts
        
        for b in range(batchsize):
            """
            for i in range(grid_h):
                for j in range(grid_w):
                    for n in range(n_box):
                        
                        best_iou = best_ious[b,n,0,i,j]
                        best_t = best_ts[b,n,0,i,j]
                        
                        if best_iou <= self.ignore_thresh:
                            tconf[b,n,0,i,j] = 0
                        if best_iou > self.truth_thresh:
                            tconf[b,n,0,i,j] = 1
                            label = tlabel[b][best_t]
                            tprob[b,n,:,i,j] = 0
                            tprob[b,n,label,i,j] = 1
                            
                            truth = tbox[b][best_t]

                            tx[b,n,0,i,j] = truth[0]*grid_w-j
                            ty[b,n,0,i,j] = truth[1]*grid_h-i
                            tw[b,n,0,i,j] = xp.log(truth[2]/abs_anchors[n,0])
                            th[b,n,0,i,j] = xp.log(truth[3]/abs_anchors[n,1])
                            box_learning_scale[b,n,0,i,j] = 2-truth[2]*truth[3]
            """
            for t in range(len(tbox[b])):
                truth = tbox[b][t]
                best_iou = 0
                #best_n = -1
                best_n = 0
                i = xp.array(int(truth[1] * grid_h))
                j = xp.array(int(truth[0] * grid_w))
                for n in range(len(self.anchors)):
                    iou = box_iou([0,0,abs_anchors[n][0],abs_anchors[n][1]],
                                  [0,0,truth[2],truth[3]])
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                
                mask_n = -1
                for n in range(len(self.mask)):
                    if best_n == self.mask[n]:
                        mask_n = n
                        break
                
                if mask_n >= 0:
                    tx[b,mask_n,0,i,j] = truth[0]*grid_w-j
                    ty[b,mask_n,0,i,j] = truth[1]*grid_h-i
                    tw[b,mask_n,0,i,j] = xp.log(truth[2]/abs_anchors[best_n,0])
                    th[b,mask_n,0,i,j] = xp.log(truth[3]/abs_anchors[best_n,1])
                    box_learning_scale[b,mask_n,0,i,j] = (2-truth[2]*truth[3])
                    
                    tconf[b,mask_n,0,i,j] = 1
                    
                    label = tlabel[b][t]
                    tprob[b,mask_n,:,i,j] = 0
                    tprob[b,mask_n,label,i,j] = 1
                    """
                    if tprob[b,mask_n,label,i,j] in [0, 1]:
                        tprob[b,mask_n,label,i,j] = 1
                    else:
                        tprob[b,mask_n,:,i,j] = 0
                        tprob[b,mask_n,label,i,j] = 1
                    """

                    
        return tx, ty, tw, th, tconf, tprob, box_learning_scale
    
    def get_loss(self, x, tbox, tlabel, input_size):
        x, y, w, h, conf, prob = self._split(x)
        with cuda.get_device_from_array(x.data):
            tx, ty, tw, th, tconf, tprob, box_learning_scale \
            = self.get_t_and_scale(x, y, w, h, conf, prob, tbox, tlabel, input_size)
        
        x_loss = F.sum((tx - x) ** 2 * box_learning_scale) / 2
        y_loss = F.sum((ty - y) ** 2 * box_learning_scale) / 2
        w_loss = F.sum((tw - w) ** 2 * box_learning_scale) / 2
        h_loss = F.sum((th - h) ** 2 * box_learning_scale) / 2
        c_loss = F.sum((tconf - conf) ** 2) / 2
        p_loss = F.sum((tprob - prob) ** 2) / 2
        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        loss = loss / x.shape[0]
        #print("x", x_loss.data, "y", y_loss.data, "w", w_loss.data, "h", h_loss.data, "c", c_loss.data, "p", p_loss.data, "loss", loss.data)
        return loss
    
    def get_detection(self, x, input_size, thresh=0.6):
        x = self._split(x)
        x = [cuda.to_cpu(e.data) for e in x]
        x, y, w, h, conf, prob = x
        batchsize, n_box, _, grid_h, grid_w = x.shape
        
        return _detection(
            x, y, w, h, conf, prob,
            batchsize, n_box, grid_h, grid_w,
            np.array(input_size, dtype=np.int32), thresh, 
            np.array(self.anchors, dtype=np.int32), 
            np.array(self.mask, dtype=np.int32)
        )
        
    """
    def get_detection(self, x, input_size, thresh=0.6):
        x = self._split(x)
        x = [cuda.to_cpu(e.data) for e in x]
        x, y, w, h, conf, prob = x
        batchsize, n_box, _, grid_h, grid_w = x.shape
        
        detections = []
        for b in range(batchsize):
            dets = []
            for i in range(grid_h):
                for j in range(grid_w):
                    for n in range(n_box):
                        
                        if conf[b,n,0,i,j] <= thresh:
                            continue
                        
                        det = {
                            'conf': conf[b,n,0,i,j],
                            'box': [
                                (j + x[b,n,0,i,j]) / grid_w,
                                (i + y[b,n,0,i,j]) / grid_h,
                                np.exp(w[b,n,0,i,j]) * self.anchors[self.mask[n]][0] / input_size[0],
                                np.exp(h[b,n,0,i,j]) * self.anchors[self.mask[n]][1] / input_size[1]
                            ],
                            'prob': prob[b,n,:,i,j]
                        }
                        if not np.all(~np.isnan(det['box'])):
                            continue
                        dets.append(det)
                        
            detections.append(dets)
        
        return detections
    """