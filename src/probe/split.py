from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..store import ActivationDataset


def filter_divergent_tasks(
    dataset: ActivationDataset,
    task_col: str = "task_id",
) -> ActivationDataset:
    """
    Keeps only tasks that exhibit outcome divergence across rollouts
    (i.e., tasks having both at least one success and at least one failure).

    This isolates task-controlled variation for probing, ensuring the probe
    learns execution signals rather than memorizing task difficulty.
    """
    if task_col not in dataset.metadata.columns:
        raise ValueError(f"Grouping column '{task_col}' not found in dataset metadata.")

    task_stats = dataset.metadata.groupby(task_col)["success"].agg(["min", "max"])
    divergent_tasks = task_stats[(task_stats["min"] == 0) & (task_stats["max"] == 1)].index

    divergent_mask = dataset.metadata[task_col].isin(divergent_tasks).values
    matching_indices = np.where(divergent_mask)[0]

    if len(matching_indices) == 0:
        raise ValueError("No divergent tasks found with both successes and failures.")

    return dataset.slice(matching_indices)


def rebalance_dataset(
    dataset: ActivationDataset,
    ratio: float,
    seed: int = 42,
) -> ActivationDataset:
    """
    Resamples the dataset to achieve a target ratio of successes vs failures
    (e.g., ratio=0.5 for 50% success and 50% failure).

    Downsamples the majority class without replacement to match the desired ratio.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"target_success_ratio must be between 0.0 and 1.0, got {ratio}")

    rng = np.random.default_rng(seed)
    pos_indices = np.where(dataset.y == 1)[0]
    neg_indices = np.where(dataset.y == 0)[0]
    num_pos = len(pos_indices)
    num_neg = len(neg_indices)

    if num_pos == 0 or num_neg == 0:
        raise ValueError(f"Cannot rebalance: dataset has {num_pos} successes and {num_neg} failures.")

    neg_needed = int(round(num_pos * (1.0 - ratio) / ratio))
    if neg_needed <= num_neg:
        chosen_neg = rng.choice(neg_indices, size=neg_needed, replace=False)
        chosen_pos = pos_indices
    else:
        pos_needed = int(round(num_neg * ratio / (1.0 - ratio)))
        chosen_pos = rng.choice(pos_indices, size=pos_needed, replace=False)
        chosen_neg = neg_indices

    selected = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(selected)
    return dataset.slice(selected)


def split_activation_dataset(
    dataset: ActivationDataset,
    eval_size: float = 0.8,
    group_by: Union[str, list[str]] = "task_id",
    divergent_tasks_only: bool = False,
    target_success_ratio: Optional[float] = None,
    seed: int = 42,
) -> tuple[ActivationDataset, ActivationDataset]:
    """
    Splits an ActivationDataset into train and eval subsets:
    1. Optionally filters to divergent tasks (tasks with both success and failure across rollouts).
    2. Performs group-aware splitting to prevent task/trajectory leakage.
    3. Optionally rebalances the training set to a target success percentage.
    """
    working_dataset = dataset

    # Resolve grouping column
    group_col = group_by[0] if isinstance(group_by, (list, tuple)) else str(group_by)
    if group_col not in working_dataset.metadata.columns:
        alt_col = "task_idx" if group_col == "task_id" else "task_id"
        if alt_col in working_dataset.metadata.columns:
            group_col = alt_col

    # 1. Filter to divergent tasks if requested
    if divergent_tasks_only:
        working_dataset = filter_divergent_tasks(working_dataset, task_col=group_col)

    # 2. Group-aware train / eval split to prevent leakage
    if group_col in working_dataset.metadata.columns:
        groups = working_dataset.metadata[group_col].values
        splitter = GroupShuffleSplit(n_splits=1, test_size=eval_size, random_state=seed)
        train_indices, eval_indices = next(splitter.split(working_dataset.X, working_dataset.y, groups=groups))
    else:
        all_indices = np.arange(working_dataset.num_samples)
        train_indices, eval_indices = train_test_split(
            all_indices,
            test_size=eval_size,
            stratify=working_dataset.y if len(np.unique(working_dataset.y)) > 1 else None,
            random_state=seed,
        )

    train_dataset = working_dataset.slice(train_indices)
    eval_dataset = working_dataset.slice(eval_indices)

    # 3. Rebalance training set class ratio if requested
    if target_success_ratio is not None:
        train_dataset = rebalance_dataset(train_dataset, ratio=target_success_ratio, seed=seed)

    return train_dataset, eval_dataset
