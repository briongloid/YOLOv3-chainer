import numpy as np

def correct_boxes(detections, original_size, input_size):
    ow, oh = original_size
    iw, ih = input_size
    new_w=0
    new_h=0
    if iw/ow < ih/oh:
        new_w = iw
        new_h = (oh * iw)/ow
    else:
        new_h = ih
        new_w = (ow * ih)/oh
    
    for i in range(len(detections)):
        b = detections[i]['box']
        b[0] =  (b[0] - (iw - new_w)/2./iw) / (new_w/iw)
        b[1] =  (b[1] - (ih - new_h)/2./ih) / (new_h/ih) 
        b[2] *= iw/new_w
        b[3] *= ih/new_h
        detections[i]['box'] = b

class YOLOv3Predictor(object):
    
    def __init__(self, predictor, thresh=0.5):
        self.predictor = predictor
        self.thresh = thresh
    
    def __call__(self, x, original_size=None):
        input_size = x.shape[2:4][::-1]
        
        correct = False
        if original_size is not None and original_size != input_size:
            correct = True
        
        ys = self.predictor(x)
        
        detections = [[] for _ in range(len(x))]
        
        for y in ys:
            dets = y['layer'].get_detection(y['y'], input_size, self.thresh)
            
            for b in range(len(x)):
                for det in dets[b]:
                    det['box'] = np.array(det['box'], dtype=np.float32)
                    det['box'] = np.clip(det['box'], -100000, 100000)
                    det['conf'] = np.array(det['conf'], dtype=np.float32)
                    prob = np.array(det['prob'], dtype=np.float32)
                    prob *= det['conf']
                    prob[prob<self.thresh] = 0
                    det['prob'] = prob
                
                detections[b] += dets[b]
        
        if correct:
            for b in range(len(x)):
                correct_boxes(detections[b], original_size, input_size)
        
        return detections
