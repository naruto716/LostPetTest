"""
Logger utility for Dog ReID training.
Based on CLIP-ReID's logging approach.
"""

import logging
import os
import sys

def setup_logger(name, save_dir=None, if_train=True):
    """
    Setup logger for training or testing.
    
    Args:
        name: Logger name
        save_dir: Directory to save log files
        if_train: Whether this is for training (affects log level)
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if save_dir is provided
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = "train.log" if if_train else "test.log"
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
