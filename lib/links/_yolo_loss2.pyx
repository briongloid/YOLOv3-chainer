
import numpy as np
#cimport numpy as np

from chainer.backends import cuda

cdef float _overlap(float x1, float len1, float x2, float len2):
    cdef float len1_half = len1/2
    cdef float len2_half = len2/2
    
    cdef float left = max(x1 - len1_half, x2 - len2_half)
    cdef float right = max(x1 + len1_half, x2 + len2_half)
    
    return right - left

cdef float _box_iou(float[4] a, float[4] b):
    
    cdef float w = _overlap(a[0], a[2], b[0], b[2])
    cdef float h = _overlap(a[1], a[3], b[1], b[3])
    
    w = max(w, 0)
    h = max(h, 0)
    
    cdef float area_i = w*h
    cdef float area_a = a[2]*a[3]
    cdef float area_b = b[2]*b[3]
    return area_i / (area_a+area_b-area_i)

def _t_and_scale(x, y, w, h, 
                 tx, ty, tw, th, tconf, tprob,
                 box_learning_scale,
                 int batchsize, int n_box, 
                 int grid_h, int grid_w,
                 input_size, anchors, abs_anchors,
                 int[:] mask, float ignore_thresh,
                 tbox, tlabel, xp):
    
    cdef int b
    cdef int i
    cdef int j
    cdef int n
    cdef int t
    
    cdef float[4] pred
    cdef float[4] truth
    cdef float iou
    cdef float best_iou
    cdef int best_t
    cdef int best_n
    
    cdef int n_tbox
    cdef int n_anchor = len(anchors)
    cdef int n_mask = len(mask)
    
    for b in range(batchsize):
        n_tbox = len(tbox[b])
        for i in range(grid_h):
            for j in range(grid_w):
                for n in range(n_box):
                    pred = [
                        (j + x[b,n,0,i,j]) / grid_w,
                        (i + y[b,n,0,i,j]) / grid_h,
                        xp.exp(w[b,n,0,i,j]) * anchors[mask[n]][0] / input_size[0],
                        xp.exp(h[b,n,0,i,j]) * anchors[mask[n]][1] / input_size[1]
                    ]
                        
                    best_iou = 0.0
                    best_t = 0
                    
                    for t in range(n_tbox):
                        truth = tbox[b][t]
                        iou = _box_iou(pred, truth)
                        if iou > best_iou:
                            best_iou = iou
                            best_t = t
                    
                    if best_iou <= ignore_thresh:
                        tconf[b,n,0,i,j] = 0
        
        for t in range(n_tbox):
            truth = tbox[b][t]
            best_iou = 0.0
            best_n = 0
            i = int(truth[1]*grid_h)
            j = int(truth[0]*grid_w)
            
            for n in range(n_anchor):
                iou = _box_iou([0,0,abs_anchors[n][0],abs_anchors[n][1]],
                                  [0,0,truth[2],truth[3]])
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            mask_n = -1
            for n in range(n_mask):
                if best_n == mask[n]:
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
                
    return tx, ty, tw, th, tconf, tprob, box_learning_scale