# Training processor for Dog ReID

from .processor_dogreid import do_train, do_inference
from .processor_regional import do_train as do_train_regional, do_inference as do_inference_regional

__all__ = ['do_train', 'do_inference', 'do_train_regional', 'do_inference_regional']
