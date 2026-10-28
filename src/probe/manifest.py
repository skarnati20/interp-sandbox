"""Saved group assignments for comparing probes across models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _group_keys(metadata: pd.DataFrame, columns: list[str]) -> list[str]:
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(column, str) for column in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("group_by must contain distinct column names")
    missing = set(columns) - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing grouping columns: {sorted(missing)}")
    if metadata[columns].isna().any().any():
        raise ValueError("Grouping columns must not contain missing values")
    return [json.dumps(values, allow_nan=False) for values in metadata[columns].itertuples(index=False, name=None)]


def _assignments(manifest: dict) -> dict[str, str]:
    columns = manifest["group_by"]
    assignments = {}
    for split in ("train", "test"):
        groups = manifest[split]
        if any(not isinstance(group, list) or len(group) != len(columns) for group in groups):
            raise ValueError("Manifest groups must match group_by columns")
        for key in _group_keys(pd.DataFrame(groups, columns=columns), columns):
            if key in assignments:
                raise ValueError(f"Duplicate or conflicting assignment for group {key}")
            assignments[key] = split
    return assignments


def create_split_manifest(
    metadata: pd.DataFrame,
    eval_size: float = 0.8,
    group_by: str | list[str] = "task_id",
    seed: int = 42,
) -> dict:
    """Split the full task/group catalog once, before filtering individual models.

    Pass existing activation metadata or a DataFrame of intended task IDs.
    Use multiple columns to namespace IDs, or a goal column for goal holdouts.
    Save the result and reuse it; a seed alone cannot handle missing tasks.

        manifest = create_split_manifest(all_tasks, eval_size=0.2)
        save_split_manifest(manifest, "split.json")
    """
    columns = [group_by] if isinstance(group_by, str) else group_by
    groups = sorted(set(_group_keys(metadata, columns)))
    train, test = train_test_split(groups, test_size=eval_size, random_state=seed)
    return {"group_by": list(columns), "train": [json.loads(key) for key in train],
            "test": [json.loads(key) for key in test]}


def save_split_manifest(manifest: dict, path: str | Path) -> None:
    """Save assignments without replacing an existing manifest."""
    _assignments(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, allow_nan=False)
        fp.write("\n")


def load_split_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fp:
        manifest = json.load(fp)
    _assignments(manifest)
    return manifest


def manifest_split_indices(metadata: pd.DataFrame, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved assignments to any subset; reject groups outside the catalog."""
    assignments = _assignments(manifest)
    keys = _group_keys(metadata, manifest["group_by"])
    unknown = set(keys) - assignments.keys()
    if unknown:
        raise ValueError(f"Groups missing from manifest: {sorted(unknown)}")
    splits = np.array([assignments[key] for key in keys])
    return np.flatnonzero(splits == "train"), np.flatnonzero(splits == "test")
