"""Project-wide utilities."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 312):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)