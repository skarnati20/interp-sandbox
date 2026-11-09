from __future__ import annotations

from typing import Optional
import numpy as np

from ...store import ActivationDataset
from ..base import Steerer


class CentroidSteerer(Steerer):
    """
    Computes a steering vector from the difference between class centroids:
        direction = mean(X[y == 1]) - mean(X[y == 0])
    Intervention shifts activations along the direction scaled by alpha:
        activation' = activation + alpha * direction
    """

    def __init__(self, alpha: float = 1.0, normalize: bool = False):
        self.alpha = alpha
        self.normalize = normalize
        self.steering_direction: Optional[np.ndarray] = None

    def fit(self, train_set: ActivationDataset) -> None:
        x = train_set.X
        y = np.asarray(train_set.y)

        pos_mask = (y == 1)
        neg_mask = (y == 0)

        if not np.any(pos_mask):
            raise ValueError("CentroidSteerer requires at least one positive (class 1) sample.")
        if not np.any(neg_mask):
            raise ValueError("CentroidSteerer requires at least one negative (class 0) sample.")

        pos_centroid = np.mean(x[pos_mask], axis=0)
        neg_centroid = np.mean(x[neg_mask], axis=0)

        direction = pos_centroid - neg_centroid
        if self.normalize:
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm

        self.steering_direction = direction

    def get_vector(self) -> np.ndarray:
        if self.steering_direction is None:
            raise ValueError("CentroidSteerer has not been fitted yet. Call fit() first.")
        return self.alpha * self.steering_direction

    def intervene(self, activation: np.ndarray) -> np.ndarray:
        if self.steering_direction is None:
            raise ValueError("CentroidSteerer has not been fitted yet. Call fit() first.")
        return activation + self.alpha * self.steering_direction
