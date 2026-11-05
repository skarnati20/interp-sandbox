from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...store import ActivationDataset
from ..base import Probe

if TYPE_CHECKING:
    from ...steering.strategy import ResamplingSteeringStrategy, VectorSteeringStrategy


class LinearProbe(Probe):
    """
    Standard linear probe (Logistic Regression with L2 regularization)
    trained on residual-stream activations.
    """

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(C=c_reg, max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, train_set: ActivationDataset) -> LinearProbe:
        raw_x = train_set.X
        if raw_x.ndim == 3:
            raw_x = raw_x.reshape(raw_x.shape[0], -1)

        scaled_x = self.scaler.fit_transform(raw_x)
        self.classifier.fit(scaled_x, train_set.y)
        self.is_fitted = True
        return self

    def predict(
        self,
        eval_set: ActivationDataset | np.ndarray,
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

    def get_steerer(
        self,
        coeff: float = 1.0,
        threshold: float = 0.5,
        condition: str = "threshold_gated",
        injection_layer: Optional[int] = None,
    ) -> VectorSteeringStrategy:
        """
        Returns a configured VectorSteeringStrategy using this probe's learned
        steering direction vector.

        Args:
            coeff: Steering strength multiplier.
            threshold: Probability threshold below which steering triggers (when condition='threshold_gated').
            condition: 'threshold_gated' (intervene on failures) or 'always'.
            injection_layer: Optional layer override to inject vector.
        """
        from ...steering.strategy import VectorSteeringStrategy

        return VectorSteeringStrategy(
            steering_vector=self.steering_vector,
            coeff=coeff,
            condition=condition,
            threshold=threshold,
            injection_layer=injection_layer,
        )

    def get_resampler(
        self,
        threshold: float = 0.5,
        max_attempts: int = 5,
        resample_temperature: float = 0.7,
        mode: str = "first_above_threshold",
    ) -> ResamplingSteeringStrategy:
        """
        Returns a configured ResamplingSteeringStrategy that uses this probe
        for real-time rejection sampling and verification.
        """
        from ...steering.strategy import ResamplingSteeringStrategy

        return ResamplingSteeringStrategy(
            threshold=threshold,
            max_attempts=max_attempts,
            resample_temperature=resample_temperature,
            mode=mode,
        )

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
