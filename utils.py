import os
import torch
import logging
import argparse
from timm.utils import AverageMeter

class argparse_namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_option():
    parser = argparse.ArgumentParser('Unified Training Script', add_help=False)
    parser.add_argument('--model_to_run', type=str, required=True, help='Model name to run (e.g., as_mlp_tiny, deit_tiny, resnext50_local)')
    args = parser.parse_args()
    return args

def create_logger(output_dir, name):
    # Setup logging to both console and file
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create formatters
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='a')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger
