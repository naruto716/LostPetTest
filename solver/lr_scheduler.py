"""
Learning rate schedulers for Dog ReID.
Based on CLIP-ReID's proven WarmupMultiStepLR implementation.
"""

from bisect import bisect_right
import torch

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Multi-step learning rate scheduler with warmup.
    
    This scheduler:
    1. Linearly increases LR from warmup_factor to 1.0 during warmup
    2. Decreases LR by gamma at each milestone after warmup
    
    Args:
        optimizer: Optimizer to schedule
        milestones: List of epoch indices for LR decay
        gamma: Multiplicative factor of LR decay
        warmup_factor: LR factor at the beginning of warmup  
        warmup_iters: Number of warmup iterations
        warmup_method: Warmup method ('linear' or 'constant')
        last_epoch: Index of last epoch
    """
    
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )
            
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        warmup_factor = 1
        
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
