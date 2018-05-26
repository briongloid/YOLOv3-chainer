#cython: boundscheck=False
#cython: wraparound=False
 
import numpy as np
#cimport numpy as np
cimport cython

from cpython cimport array

def _detection(float[:,:,:,:,:] x, float[:,:,:,:,:] y, 
               float[:,:,:,:,:] w, float[:,:,:,:,:] h,
               float[:,:,:,:,:] conf, float[:,:,:,:,:] prob, 
               int batchsize, int n_box, int grid_h, int grid_w,
               int[:] input_size, float thresh, int[:,:] anchors, int[:] mask):

    cdef int b
    cdef int i
    cdef int j
    cdef int n
    
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
                            np.exp(w[b,n,0,i,j]) * anchors[mask[n]][0] / input_size[0],
                            np.exp(h[b,n,0,i,j]) * anchors[mask[n]][1] / input_size[1]
                        ],
                        'prob': prob[b,n,:,i,j]
                    }
                    dets.append(det)

        detections.append(dets)

    return detections