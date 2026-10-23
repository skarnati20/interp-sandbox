from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from ..store import ActivationDataset
from .base import Probe
from .split import split_activation_dataset


def filter_by_turn(
    dataset: ActivationDataset,
    max_turn: int,
    turn_col: str = "round",
) -> ActivationDataset:
    """
    Filters dataset to include only the first max_turn turns (e.g., round < max_turn).
    """
    col = turn_col
    if col not in dataset.metadata.columns:
        alt_col = "step_idx" if col == "round" else "round"
        if alt_col in dataset.metadata.columns:
            col = alt_col
        else:
            return dataset

    mask = (dataset.metadata[col].astype(int) < max_turn).values
    matching_indices = np.where(mask)[0]
    if len(matching_indices) == 0:
        raise ValueError(f"No samples found with turn < {max_turn} in column '{col}'.")
    return dataset.slice(matching_indices)


def train(
    probe: Probe,
    train_set: ActivationDataset,
    max_turn: Optional[int] = None,
) -> Probe:
    """
    Fits the probe on the training dataset and returns the trained probe.
    If max_turn is provided, only trains on the first max_turn turns (e.g. round < max_turn).
    """
    working_set = filter_by_turn(train_set, max_turn) if max_turn is not None else train_set
    probe.fit(working_set)
    return probe


def evaluate(
    probe: Probe,
    eval_set: ActivationDataset,
    max_turn: Optional[int] = None,
) -> dict[str, float]:
    """
    Evaluates the trained probe on an evaluation dataset.
    If max_turn is provided, only evaluates on the first max_turn turns.
    Returns a dictionary of classification metrics (accuracy, roc_auc, f1, precision, recall).
    """
    working_set = filter_by_turn(eval_set, max_turn) if max_turn is not None else eval_set
    return probe.evaluate(working_set)


def train_and_evaluate(
    probe: Probe,
    dataset: Optional[ActivationDataset] = None,
    train_set: Optional[ActivationDataset] = None,
    eval_set: Optional[ActivationDataset] = None,
    max_turn: Optional[int] = None,
    train_max_turn: Optional[int] = None,
    eval_max_turn: Optional[int] = None,
    eval_size: float = 0.8,
    group_by: Union[str, list[str]] = "task_id",
    divergent_tasks_only: bool = False,
    target_success_ratio: Optional[float] = None,
    seed: int = 42,
    *,
    manifest: Optional[dict | str | Path] = None,
    **split_options,
) -> dict[str, Any]:
    """
    Complete pipeline to split data, train probe, and evaluate performance.

    If train_set and eval_set are not directly provided, splits the given dataset
    using split_activation_dataset according to the specified controls.

    Turn cutoffs:
    - max_turn: Applies cutoff to both train and eval.
    - train_max_turn: Specific cutoff for training (overrides max_turn).
    - eval_max_turn: Specific cutoff for evaluation (overrides max_turn).

    Pass manifest to reuse saved task assignments across models. Other manifest
    options (e.g. task_catalog, eval_split) are forwarded to the shared splitter.
    """
    if (manifest is not None or split_options) and (train_set is not None or eval_set is not None):
        raise ValueError("Manifest options require dataset, without pre-split train/eval inputs")
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
            manifest=manifest,
            **split_options,
        )

    actual_train_cutoff = train_max_turn if train_max_turn is not None else max_turn
    actual_eval_cutoff = eval_max_turn if eval_max_turn is not None else max_turn

    # 1. Train probe on train_set (with optional turn cutoff)
    train(probe, train_set, max_turn=actual_train_cutoff)

    # 2. Evaluate on eval_set and train_set (with optional turn cutoffs)
    eval_metrics = evaluate(probe, eval_set, max_turn=actual_eval_cutoff)
    train_metrics = evaluate(probe, train_set, max_turn=actual_train_cutoff)

    effective_train_set = filter_by_turn(train_set, actual_train_cutoff) if actual_train_cutoff is not None else train_set
    effective_eval_set = filter_by_turn(eval_set, actual_eval_cutoff) if actual_eval_cutoff is not None else eval_set

    return {
        "probe": probe,
        "eval_metrics": eval_metrics,
        "train_metrics": train_metrics,
        "train_samples": effective_train_set.num_samples,
        "eval_samples": effective_eval_set.num_samples,
    }
