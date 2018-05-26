from chainer import training

class DarknetShift(training.Extension):
    
    def __init__(self, optimizer, policy, max_iteration, 
                 burn_in=0, power=4, steps=None, scales=None):
        self._optimizer = optimizer
        self._init = optimizer.lr
        self._policy = policy
        self._max_iteration = max_iteration
        self._t = 0
        
        self._burn_in = burn_in
        self._power = power
        self._steps = steps
        self._scales = scales
    
    def __call__(self, trainer):
        self._t += 1
        lr = self._get_lr()
        self._optimizer.lr = lr

            
    def _get_lr(self):
        n_iteration = self._t
        lr = self._init
        if n_iteration > self._max_iteration:
            n_iteration = self._max_iteration - 1
        
        if n_iteration < self._burn_in:
            return lr * pow(n_iteration / self._burn_in, self._power)
        
        if 'constant' == self._policy:
            return lr
        elif 'steps' == self._policy:
            for i in range(len(self._steps)):
                if self._steps[i] > n_iteration:
                    return lr
                lr *= self._scales[i]
            return lr
        elif 'poly' == self._policy:
            return lr * pow(1 - n_iteration/self._max_iteration, self._power)
        else:
            raise Exception()