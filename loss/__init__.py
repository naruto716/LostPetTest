# Loss functions for Dog ReID

from .triplet import BatchHardTripletLoss
from .center import CenterLoss  
from .id_loss import LabelSmoothingCrossEntropy
from .make_loss import ReIDLossBuilder, make_loss

__all__ = ['BatchHardTripletLoss', 'CenterLoss', 'LabelSmoothingCrossEntropy', 'ReIDLossBuilder', 'make_loss']
