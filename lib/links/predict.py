import numpy as np

class YOLOv3Predictor(object):
    
    def __init__(self, predictor, thresh=0.6):
        self.predictor = predictor
        self.thresh = thresh
    
    def __call__(self, x):
        input_size = x.shape[2:4][::-1]
        ys = self.predictor(x)
        
        detections = [[] for _ in range(len(x))]
        
        for y in ys:
            dets = y['layer'].get_detection(y['y'], input_size, self.thresh)
            
            for b in range(len(x)):
                for det in dets[b]:
                    det['box'] = np.array(det['box'], dtype=np.float32)
                    det['conf'] = np.array(det['conf'], dtype=np.float32)
                    prob = np.array(det['prob'], dtype=np.float32)
                    prob *= det['conf']
                    prob[prob<self.thresh] = 0
                    det['prob'] = prob
            
            for b in range(len(x)):
                detections[b] += dets[b]
        
        return detections
