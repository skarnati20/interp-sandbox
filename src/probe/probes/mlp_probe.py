from __future__ import annotations

from typing import Union
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ...store import ActivationDataset
from ..base import Probe


class MLPProbe(Probe):
    """
    Multi-Layer Perceptron (non-linear) probe with feature scaling.
    Tests whether non-linear combinations of activations yield higher predictive power.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        alpha: float = 0.01,
        max_iter: int = 500,
        early_stopping: bool = True,
        seed: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.seed = seed
        self.scaler = StandardScaler()
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=seed,
        )
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
        if not self.is_fitted:
            raise RuntimeError("MLPProbe must be fitted before calling predict.")

        raw_x = eval_set.X if hasattr(eval_set, "X") else eval_set
        scaled_x = self.scaler.transform(raw_x)

        if len(self.classifier.classes_) == 1:
            only_class = self.classifier.classes_[0]
            prob_one = 1.0 if only_class == 1 else 0.0
            probs = np.full(raw_x.shape[0], prob_one)
        else:
            probs = self.classifier.predict_proba(scaled_x)[:, 1]

        if return_proba:
            return probs
        return (probs >= threshold).astype(int)
