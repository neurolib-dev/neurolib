import numpy as np


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
