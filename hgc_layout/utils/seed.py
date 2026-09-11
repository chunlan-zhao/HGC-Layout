"""Central random-seed control for numpy and torch."""
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> np.random.Generator:
    """Seed Python, NumPy and torch; return a local NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
