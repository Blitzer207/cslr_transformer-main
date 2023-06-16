"""
This file contains a simple script to free the GPU memory during training.
"""
import torch
from GPUtil import showUtilization as gpu_usage
import gc


def free_gpu_cache():
    """
    Free GPU memory
    :return:
    """
    print("Initial GPU Usage")
    gpu_usage()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU Usage after emptying the cache")
    gpu_usage()
