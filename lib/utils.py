import numpy as np

def rand_scale(f):
    scale = np.random.uniform(1, f)
    return scale if np.random.rand()<0.5 else 1./scale

def label2onehot(label, n_class):
    return np.eye(n_class)[label]

def xyminmax2yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def yolo2xyminmax(size, box):
    w, h = size
    xmin = int(round(w*(box[0] - box[2]/2))) + 1
    xmax = int(round(w*(box[0] + box[2]/2))) + 1
    ymin = int(round(h*(box[1] - box[3]/2))) + 1
    ymax = int(round(h*(box[1] + box[3]/2))) + 1
    return (xmin, xmax, ymin, ymax)
