from __future__ import annotations

import numpy as np

from ...store import ActivationDataset
from .linear_probe import LinearProbe


class TemporalProbe(LinearProbe):
    """Linear probe over current state, causal running mean, and one-step delta.

    Uses the same scaler, classifier, and success-label convention as LinearProbe.
    Inputs must contain complete episode prefixes starting at turn zero; raw
    arrays cannot identify episode boundaries. Predictions retain input row order.
    No history is retained between calls: supply the full prefix at inference.
    """

    def temporal_features(self, dataset: ActivationDataset) -> np.ndarray:
        if not isinstance(dataset, ActivationDataset):
            raise TypeError("TemporalProbe requires an ActivationDataset with episode metadata")
        x = np.asarray(dataset.X, dtype=np.float64)
        meta = dataset.metadata.reset_index(drop=True)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("Select one activation layer to provide a 2D feature matrix")
        if len(meta) != len(x) or len(dataset.y) != len(x):
            raise ValueError("Features, labels, and metadata must have matching row counts")
        if not np.isfinite(x).all():
            raise ValueError("Activations must be finite")
        turn_col = "round" if "round" in meta else "step_idx"
        required = ["episode_id", turn_col]
        if any(column not in meta for column in required):
            raise ValueError("TemporalProbe requires episode_id and round or step_idx")
        # Separate anchors and benchmark namespaces when supplied by the dataset.
        groups = [column for column in ("benchmark", "task_id", "episode_id", "anchor")
                  if column in meta]
        if meta[groups + [turn_col]].isna().any().any():
            raise ValueError("Episode identifiers and turns must not contain missing values")
        turns = meta[turn_col].to_numpy(dtype=np.float64)
        if not np.isfinite(turns).all() or np.any(turns != np.floor(turns)):
            raise ValueError("Turns must be finite integers")
        features = np.empty((len(x), x.shape[1] * 3), dtype=np.float64)
        for indices in meta.groupby(groups, sort=False, observed=True).indices.values():
            ordered = indices[np.argsort(turns[indices], kind="stable")]
            if not np.array_equal(turns[ordered], np.arange(len(ordered))):
                raise ValueError("Each episode/anchor must be a contiguous prefix starting at turn 0; "
                                 "duplicate or missing turns are not supported")
            current = x[ordered]
            mean = np.cumsum(current, axis=0) / np.arange(1, len(ordered) + 1)[:, None]
            delta = np.zeros_like(current)
            delta[1:] = np.diff(current, axis=0)
            features[ordered] = np.concatenate([current, mean, delta], axis=1)
        return features

    def fit(self, train_set: ActivationDataset) -> None:
        features = self.temporal_features(train_set)
        scaled_x = self.scaler.fit_transform(features)
        self.classifier.fit(scaled_x, train_set.y)

    def predict(
        self,
        eval_set: ActivationDataset,
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        features = self.temporal_features(eval_set)
        return super().predict(features, return_proba=return_proba, threshold=threshold)
