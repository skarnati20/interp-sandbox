"""
Base interfaces and data structures for representation probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np

from ..store import ActivationDataset


class Probe(ABC):
    """
    Abstract base class for residual-stream probes.
    """

    @abstractmethod
    def fit(self, train_set: ActivationDataset) -> Probe:
        """Trains the probe on an ActivationDataset."""
        pass

    @abstractmethod
    def predict(
        self,
        eval_set: ActivationDataset | np.ndarray,
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predicts binary outcomes or class-1 probabilities.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Saves the trained probe model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Probe:
        """Loads a trained probe model from disk."""
        pass

    def evaluate(self, eval_set: ActivationDataset) -> dict[str, float]:
        """Calculates classification metrics on an ActivationDataset."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        probs = self.predict(eval_set, return_proba=True)
        preds = self.predict(eval_set, return_proba=False)
        y_true = eval_set.y

        metrics = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
        }

        # ROC-AUC requires at least one sample in each class
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        else:
            metrics["roc_auc"] = 0.5

        return metrics


def load_probe(path: str | Path) -> Probe:
    """
    Inspects saved probe state and dynamically dispatches to the correct Probe class loader.
    """
    state = joblib.load(path)
    probe_class = state.get("probe_class")

    if probe_class == "LinearProbe":
        from .probes.linear_probe import LinearProbe
        return LinearProbe.load(path)
    elif probe_class == "MLPProbe":
        from .probes.mlp_probe import MLPProbe
        return MLPProbe.load(path)
    elif probe_class == "XGBoostProbe":
        from .probes.xgboost_probe import XGBoostProbe
        return XGBoostProbe.load(path)

    raise ValueError(f"Unknown or missing probe_class in saved file '{path}': {probe_class}")
