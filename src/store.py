from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


class ShardWriter:
    """
    High-performance sharded storage writer.
    Consolidates activation tensors and step metadata into matching
    'feat.shard*.npz' and 'meta.shard*.jsonl' files directly compatible
    with 'Doomed from the Start' (2026).
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 2000,
        layer_ids: Optional[list[int]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.layer_ids = layer_ids
        self.shard_idx = 0

        self._current_feats: list[np.ndarray] = []
        self._current_metas: list[dict[str, Any]] = []

    def add_step(
        self,
        feat_tensor: torch.Tensor | np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        """
        Adds a single anchor activation vector and its metadata to the active shard buffer.
        Args:
            feat_tensor: Array/Tensor of shape [n_layers, hidden_dim] or [hidden_dim] (float16).
            meta: Metadata dict with {task_idx, rollout_k, round, anchor, success, ...}.
        """
        if isinstance(feat_tensor, torch.Tensor):
            feat_arr = feat_tensor.to(torch.float16).detach().cpu().numpy()
        else:
            feat_arr = np.asarray(feat_tensor, dtype=np.float16)

        if feat_arr.ndim == 1:
            feat_arr = np.expand_dims(feat_arr, axis=0)  # Shape: [1, hidden_dim]

        self._current_feats.append(feat_arr)
        self._current_metas.append(meta)

        if len(self._current_feats) >= self.shard_size:
            self.flush()

    def flush(self) -> Optional[tuple[Path, Path]]:
        if not self._current_feats:
            return None

        feat_path = self.output_dir / f"feat.shard{self.shard_idx}.npz"
        meta_path = self.output_dir / f"meta.shard{self.shard_idx}.jsonl"

        # Stack into [N, n_layers, hidden_dim]
        X = np.stack(self._current_feats, axis=0)
        layer_ids_arr = (
            np.array(self.layer_ids)
            if self.layer_ids is not None
            else np.arange(X.shape[1])
        )

        # 1. Save compressed numpy tensor shard
        np.savez_compressed(feat_path, X=X, layer_ids=layer_ids_arr)

        # 2. Save matching jsonl metadata shard
        with open(meta_path, "w", encoding="utf-8") as fp:
            for record in self._current_metas:
                fp.write(json.dumps(record) + "\n")

        self.shard_idx += 1
        self._current_feats.clear()
        self._current_metas.clear()

        return feat_path, meta_path

    def close(self) -> None:
        self.flush()


def save_run_summary(file_path: str | Path, records: list[dict[str, Any]]) -> Path:
    """Saves a list of episode result dicts to a single Parquet file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return path


def load_run_summary(file_path: str | Path) -> pd.DataFrame:
    """Loads an episode run summary Parquet file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return pd.read_parquet(path)
