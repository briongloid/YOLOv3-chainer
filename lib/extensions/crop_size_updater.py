import numpy as np
from chainer import training
from chainer.training import trigger as trigger_module

class CropSizeUpdater(training.Extension): 
    
    def __init__(self, dataset, crop_sizes, max_size_iteration=None,
                 trigger=(80, 'iteration')):
        super(CropSizeUpdater, self).__init__()
        self.dataset = dataset
        self.crop_sizes = crop_sizes
        self.max_size_iteration = max_size_iteration
        self._trigger = trigger_module.get_trigger(trigger)
    
    def __call__(self, trainer):
        
        if self.max_size_iteration is not None \
           and trainer.updater.iteration > self.max_size_iteration:
            self.dataset.set_crop_size(self.crop_sizes[-1])
            return
        
        if self._trigger(trainer):
            index = np.random.randint(len(self.crop_sizes))
            self.dataset.set_crop_size(self.crop_sizes[index])