from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


@dataclass
class StudentConfig:
    max_depth: int = 8
    min_samples_leaf: int = 10


class StudentTree:
    def __init__(self, cfg: StudentConfig, mode: str = "regression"):
        self.cfg = cfg
        self.mode = mode
        self.model = None

    def fit(self, X, y):
        params = dict(
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
        )

        if self.mode == "regression":
            self.model = DecisionTreeRegressor(**params)
        else:
            self.model = DecisionTreeClassifier(**params)

        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        raw = self.model.predict(X)

        if self.mode == "regression":
            return np.clip(np.round(raw), 1, 6).astype(int)
        else:
            return raw.astype(int)