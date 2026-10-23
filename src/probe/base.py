from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np

from ..activations import ExtractionResult
from ..store import ActivationDataset


class Probe(ABC):
    """
    Abstract base class for activation probes.
    """

    @abstractmethod
    def fit(self, train_set: ActivationDataset) -> None:
        """Trains the probe on an ActivationDataset."""
        pass

    @abstractmethod
    def predict(
        self,
        eval_set: Union[ActivationDataset, np.ndarray],
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Returns predictions for the input samples.
        If return_proba is True, returns predicted probabilities of success (class 1).
        If return_proba is False, returns binary predictions (0 or 1) based on threshold.
        """
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
            metrics["roc_auc"] = float("nan")

        return metrics
