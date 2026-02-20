import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_stratified_folds(y, n_splits=5, seed=42):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(skf.split(np.zeros(len(y)), y))