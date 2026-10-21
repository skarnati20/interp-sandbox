from __future__ import annotations

from typing import Any, Optional, Union

from ..store import ActivationDataset
from .base import Probe
from .split import split_activation_dataset


def train(
    probe: Probe,
    train_set: ActivationDataset,
) -> Probe:
    """
    Fits the probe on the training dataset and returns the trained probe.
    """
    probe.fit(train_set)
    return probe


def evaluate(
    probe: Probe,
    eval_set: ActivationDataset,
) -> dict[str, float]:
    """
    Evaluates the trained probe on an evaluation dataset.
    Returns a dictionary of classification metrics (accuracy, roc_auc, f1, precision, recall).
    """
    return probe.evaluate(eval_set)


def train_and_evaluate(
    probe: Probe,
    dataset: Optional[ActivationDataset] = None,
    train_set: Optional[ActivationDataset] = None,
    eval_set: Optional[ActivationDataset] = None,
    eval_size: float = 0.8,
    group_by: Union[str, list[str]] = "task_id",
    divergent_tasks_only: bool = False,
    target_success_ratio: Optional[float] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Complete pipeline to split data, train probe, and evaluate performance.

    If train_set and eval_set are not directly provided, splits the given dataset
    using split_activation_dataset according to the specified controls.
    """
    if train_set is None or eval_set is None:
        if dataset is None:
            raise ValueError("Must provide either 'dataset' or both 'train_set' and 'eval_set'.")
        train_set, eval_set = split_activation_dataset(
            dataset=dataset,
            eval_size=eval_size,
            group_by=group_by,
            divergent_tasks_only=divergent_tasks_only,
            target_success_ratio=target_success_ratio,
            seed=seed,
        )

    # 1. Train probe on train_set
    train(probe, train_set)

    # 2. Evaluate on eval_set and train_set
    eval_metrics = evaluate(probe, eval_set)
    train_metrics = evaluate(probe, train_set)

    return {
        "probe": probe,
        "eval_metrics": eval_metrics,
        "train_metrics": train_metrics,
        "train_samples": train_set.num_samples,
        "eval_samples": eval_set.num_samples,
    }
