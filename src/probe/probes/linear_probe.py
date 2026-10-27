from __future__ import annotations

from pathlib import Path
from typing import Union
import joblib
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
        self.is_fitted = False

    def fit(self, train_set: ActivationDataset) -> None:
        scaled_x = self.scaler.fit_transform(train_set.X)
        self.classifier.fit(scaled_x, train_set.y)
        self.is_fitted = True

    def predict(
        self,
        eval_set: Union[ActivationDataset, np.ndarray],
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        raw_x = eval_set.X if hasattr(eval_set, "X") else eval_set
        scaled_x = self.scaler.transform(raw_x)
        probs = self.classifier.predict_proba(scaled_x)[:, 1]
        if return_proba:
            return probs
        return (probs >= threshold).astype(int)

    @property
    def steering_vector(self) -> np.ndarray:
        """
        Computes the normalized unit steering direction pointing towards success (class 1)
        in the original unscaled activation space: v = w / scale.
        """
        if not hasattr(self.classifier, "coef_"):
            raise RuntimeError("LinearProbe must be fitted before extracting steering vector.")
        raw_w = self.classifier.coef_[0]
        if hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None:
            v = raw_w / np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
        else:
            v = raw_w
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 0 else v

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "probe_class": "LinearProbe",
            "c_reg": self.c_reg,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        state = joblib.load(path)
        probe = cls(c_reg=state.get("c_reg", 1.0))
        probe.scaler = state["scaler"]
        probe.classifier = state["classifier"]
        probe.is_fitted = state.get("is_fitted", True)
        return probe
