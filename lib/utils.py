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

def detection2text(bbox, conf, prob, names, thresh, allow_conf_only=False):
    text = 'conf {:.3}'.format(float(conf))
    prob_texts = []
    for i, p in enumerate(prob):
        if p >= thresh:
            prob_texts.append('{} {:.3}'.format(names[i], float(p)))
    if len(prob_texts) > 0:
        text += ':' + ','.join(prob_texts)
        return text
    else:
        if allow_conf_only:
            return text
        else:
            return ''
    
    
    