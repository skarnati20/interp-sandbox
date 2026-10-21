from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import torch

from .activations import ExtractionResult
from .hf import download_activations


@dataclass
class ActivationDataset:
    """
    Container holding extracted activation features, binary success labels,
    and aligned trajectory metadata.
    """
    X: np.ndarray              # Shape: [N, hidden_dim] or [N, n_layers, hidden_dim]
    y: np.ndarray              # Shape: [N] (binary 0/1 success labels)
    layer_ids: list[int]       # Transformer layer indices
    metadata: pd.DataFrame     # Aligned DataFrame with trajectory metadata

    @property
    def num_samples(self) -> int:
        return self.X.shape[0]

    def to_torch(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Converts X and y to PyTorch tensors on the target device."""
        return (
            torch.from_numpy(self.X).to(device),
            torch.from_numpy(self.y).to(device),
        )

    def slice(self, indices: np.ndarray) -> ActivationDataset:
        return ActivationDataset(
            X=self.X[indices],
            y=self.y[indices],
            layer_ids=self.layer_ids,
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
        )

    def __getitem__(self, indices: np.ndarray) -> ActivationDataset:
        return self.slice(indices)


class ShardWriter:
    """
    High-performance sharded storage writer.
    Consolidates activation tensors and step metadata into matching
    'feat.shard*.npz' and 'meta.shard*.jsonl' files.
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

        self.current_feats: list[np.ndarray] = []
        self.current_metas: list[dict[str, Any]] = []

    def add_step(
        self,
        feat_tensor: torch.Tensor | np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        """
        Adds a single anchor activation vector and its metadata to the active shard buffer.
        """
        if isinstance(feat_tensor, torch.Tensor):
            feat_arr = feat_tensor.to(torch.float16).detach().cpu().numpy()
        else:
            feat_arr = np.asarray(feat_tensor, dtype=np.float16)

        if feat_arr.ndim == 1:
            feat_arr = np.expand_dims(feat_arr, axis=0)  # Shape: [1, hidden_dim]

        self.current_feats.append(feat_arr)
        self.current_metas.append(meta)

        if len(self.current_feats) >= self.shard_size:
            self.flush()

    def flush(self) -> Optional[tuple[Path, Path]]:
        if not self.current_feats:
            return None

        feat_path = self.output_dir / f"feat.shard{self.shard_idx}.npz"
        meta_path = self.output_dir / f"meta.shard{self.shard_idx}.jsonl"

        # Stack into [N, n_layers, hidden_dim]
        stacked_x = np.stack(self.current_feats, axis=0)
        layer_ids_arr = (
            np.array(self.layer_ids)
            if self.layer_ids is not None
            else np.arange(stacked_x.shape[1])
        )

        # 1. Save compressed numpy tensor shard
        np.savez_compressed(feat_path, X=stacked_x, layer_ids=layer_ids_arr)

        # 2. Save matching jsonl metadata shard
        with open(meta_path, "w", encoding="utf-8") as fp:
            for record in self.current_metas:
                fp.write(json.dumps(record) + "\n")

        self.shard_idx += 1
        self.current_feats.clear()
        self.current_metas.clear()

        return feat_path, meta_path

    def close(self) -> None:
        self.flush()


class ShardReader:
    """
    Reads stored activation shards ('feat.shard*.npz' and 'meta.shard*.jsonl')
    from local storage or directly from Hugging Face repositories.
    """

    def __init__(
        self,
        data_dir: str | Path,
        token: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.token = token

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        repo_type: str = "dataset",
        allow_patterns: Optional[list[str]] = None,
        local_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
    ) -> ShardReader:
        """
        Downloads activation shards from a Hugging Face repository and returns
        a ShardReader configured to read the downloaded files.
        """
        download_dir = download_activations(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            token=token,
        )
        return cls(data_dir=download_dir, token=token)

    @classmethod
    def read_from_huggingface(
        cls,
        repo_id: str,
        anchor: Optional[str] = None,
        layer: Optional[int] = None,
        repo_type: str = "dataset",
        allow_patterns: Optional[list[str]] = None,
        local_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
    ) -> ActivationDataset:
        """
        Convenience method: downloads from Hugging Face and returns the
        consolidated ActivationDataset filtered by anchor and layer.
        """
        reader = cls.from_huggingface(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            token=token,
        )
        return reader.read_all(anchor=anchor, layer=layer)

    def get_shard_files(self) -> list[tuple[Path, Path]]:
        """Finds all matching (feat.shard*.npz, meta.shard*.jsonl) file pairs."""
        feat_files = sorted(self.data_dir.rglob("feat.shard*.npz"))
        pairs = []
        for feat_path in feat_files:
            meta_name = feat_path.name.replace("feat.", "meta.").replace(".npz", ".jsonl")
            meta_path = feat_path.with_name(meta_name)
            if meta_path.exists():
                pairs.append((feat_path, meta_path))
        return pairs

    def load_single_shard(
        self,
        feat_path: Path,
        meta_path: Path,
        anchor: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Optional[ActivationDataset]:
        """Loads and filters a single shard pair."""
        npz_data = np.load(feat_path)
        raw_x = npz_data["X"]            # Shape: [N, n_layers, hidden_dim] or [N, hidden_dim]
        layer_ids = list(npz_data["layer_ids"])
        meta_df = pd.read_json(meta_path, lines=True)

        # 1. Filter by anchor if specified
        if anchor is not None and "anchor" in meta_df.columns:
            mask = (meta_df["anchor"].astype(str).str.lower() == anchor.strip().lower()).values
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return None
            filtered_x = raw_x[indices]
            filtered_meta = meta_df.iloc[indices].reset_index(drop=True)
        else:
            filtered_x = raw_x
            filtered_meta = meta_df.reset_index(drop=True)

        # 2. Slice target layer if specified and tensor has layer dimension
        selected_layers = layer_ids
        if layer is not None and filtered_x.ndim == 3:
            if layer not in layer_ids:
                raise ValueError(f"Requested layer {layer} not found in shard layers {layer_ids}")
            layer_pos = layer_ids.index(layer)
            filtered_x = filtered_x[:, layer_pos, :]  # Shape: [N, hidden_dim]
            selected_layers = [layer]

        # 3. Extract binary success labels
        if "success" in filtered_meta.columns:
            y_labels = filtered_meta["success"].values.astype(int)
        else:
            y_labels = np.zeros(len(filtered_meta), dtype=int)

        return ActivationDataset(
            X=filtered_x,
            y=y_labels,
            layer_ids=selected_layers,
            metadata=filtered_meta,
        )

    def iter_shards(
        self,
        anchor: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Iterator[ActivationDataset]:
        """
        Yields an ActivationDataset shard-by-shard to minimize peak memory usage.
        """
        pairs = self.get_shard_files()
        for feat_path, meta_path in pairs:
            dataset = self.load_single_shard(feat_path, meta_path, anchor=anchor, layer=layer)
            if dataset is not None and dataset.num_samples > 0:
                yield dataset

    def read_all(
        self,
        anchor: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> ActivationDataset:
        """
        Loads and concatenates all shards into a single consolidated ActivationDataset.
        """
        x_list = []
        y_list = []
        meta_list = []
        final_layers: list[int] = []

        for shard_dataset in self.iter_shards(anchor=anchor, layer=layer):
            x_list.append(shard_dataset.X)
            y_list.append(shard_dataset.y)
            meta_list.append(shard_dataset.metadata)
            final_layers = shard_dataset.layer_ids

        if not x_list:
            return ActivationDataset(
                X=np.empty((0, 0), dtype=np.float16),
                y=np.empty((0,), dtype=int),
                layer_ids=[],
                metadata=pd.DataFrame(),
            )

        return ActivationDataset(
            X=np.concatenate(x_list, axis=0),
            y=np.concatenate(y_list, axis=0),
            layer_ids=final_layers,
            metadata=pd.concat(meta_list, ignore_index=True),
        )


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
