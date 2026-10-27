from __future__ import annotations

import os
# Prevent OpenMP thread conflicts on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import Union
import joblib
import numpy as np
import xgboost as xgb

from ...store import ActivationDataset
from ..base import Probe


class XGBoostProbe(Probe):
    """
    Gradient boosted decision tree probe using XGBoost.
    Evaluates tree-based non-linear partitions over activation dimensions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        colsample_bytree: float = 0.3,
        subsample: float = 0.8,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.seed = seed
        self.classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        self.is_fitted = False

    def fit(self, train_set: ActivationDataset) -> None:
        train_x = np.asarray(train_set.X, dtype=np.float32)
        train_y = np.asarray(train_set.y, dtype=int)
        self.classifier.fit(train_x, train_y)
        self.is_fitted = True

    def predict(
        self,
        eval_set: Union[ActivationDataset, np.ndarray],
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("XGBoostProbe must be fitted before calling predict.")

        raw_x = eval_set.X if hasattr(eval_set, "X") else eval_set
        eval_x = np.asarray(raw_x, dtype=np.float32)

        if len(self.classifier.classes_) == 1:
            only_class = self.classifier.classes_[0]
            prob_one = 1.0 if only_class == 1 else 0.0
            probs = np.full(eval_x.shape[0], prob_one)
        else:
            probs = self.classifier.predict_proba(eval_x)[:, 1]

        if return_proba:
            return probs
        return (probs >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "probe_class": "XGBoostProbe",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "subsample": self.subsample,
            "seed": self.seed,
            "classifier": self.classifier,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> XGBoostProbe:
        state = joblib.load(path)
        probe = cls(
            n_estimators=state.get("n_estimators", 100),
            max_depth=state.get("max_depth", 3),
            learning_rate=state.get("learning_rate", 0.05),
            colsample_bytree=state.get("colsample_bytree", 0.3),
            subsample=state.get("subsample", 0.8),
            seed=state.get("seed", 42),
        )
        probe.classifier = state["classifier"]
        probe.is_fitted = state.get("is_fitted", True)
        return probe
