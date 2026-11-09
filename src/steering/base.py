from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..store import ActivationDataset


class Steerer(ABC):
    @abstractmethod
    def fit(self, train_set: ActivationDataset) -> None:
        """Fits a steering method based off initial training data."""
        pass

    @abstractmethod
    def intervene(
        self,
        activation: np.ndarray,
    ) -> np.ndarray:
        """Given an activation, returns a new activation after intervention."""
        pass

    def get_vector(self) -> Optional[np.ndarray]:
        """Returns the steering direction vector if applicable."""
        return None
