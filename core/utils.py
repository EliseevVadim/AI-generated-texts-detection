import random
import gc
import numpy as np
import torch


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
