from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...store import ActivationDataset
from ..base import Probe


class LinearProbe(Probe):
    def __init__(self, c_reg: float = 1.0, max_iter: int = 1000):
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(C=c_reg, max_iter=max_iter)

    def fit(self, train_set: ActivationDataset) -> None:
        scaled_x = self.scaler.fit_transform(train_set.X)
        self.classifier.fit(scaled_x, train_set.y)

    def predict(
        self,
        eval_set: ActivationDataset,
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        raw_x = eval_set.X if hasattr(eval_set, "X") else eval_set
        scaled_x = self.scaler.transform(raw_x)
        probs = self.classifier.predict_proba(scaled_x)[:, 1]
        if return_proba:
            return probs
        return (probs >= threshold).astype(int)
