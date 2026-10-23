"""Load collected activations and prepare Hugging Face train/test datasets."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from datasets import Dataset, DatasetDict
from src.hf import download_activations
import numpy as np

from src.splits import (annotate_activation_metadata, create_manifest_from_metadata,
                        load_split_manifest)
from src.store import ShardReader


def _iter_activation_rows(shards: list[dict], layout: dict) -> Iterator[dict]:
    expected_shape = tuple(layout["shape"])
    expected_layers = layout["layer_ids"]

    for descriptor in shards:
        feat_path = Path(descriptor["features"])
        meta_path = Path(descriptor["metadata"])

        shard = ShardReader(feat_path.parent).load_single_shard(feat_path, meta_path)
        if shard.X.shape[1:] != expected_shape or shard.layer_ids != list(expected_layers):
            raise ValueError(f"Activation shape or layer_ids differ across shards: {feat_path}")
        for activation, meta in zip(shard.X, shard.metadata.to_dict("records")):
            if "source_provenance" in meta:
                meta["source_provenance"] = json.dumps(meta["source_provenance"], sort_keys=True)
            yield {
                **meta,
                "activations": activation.astype(np.float16, copy=False),
                "layer_ids": shard.layer_ids,
            }


def load_activation_dataset(
    data_dir: str | Path,
    cache_dir: Optional[str | Path] = None,
) -> Dataset:
    """
    Loads a run directory or its features/ directory into a Hugging Face Dataset.
    Each row contains one [n_layers, hidden_dim] float16 activation array,
    layer_ids, and the corresponding JSONL metadata (including success).
    Shards are read individually and cached as an Arrow dataset on disk.
    """
    directory = Path(data_dir).resolve()
    features_dir = directory / "features" if (directory / "features").is_dir() else directory
    pairs = ShardReader(features_dir).get_shard_files()
    if not pairs:
        raise FileNotFoundError(f"No activation shards found in {features_dir}")
    feat_paths = [features for features, _ in pairs]

    with np.load(feat_paths[0], allow_pickle=False) as first:
        if "X" not in first or "layer_ids" not in first:
            raise ValueError(f"Missing X or layer_ids in {feat_paths[0]}")
        layout = {"shape": first["X"].shape[1:], "layer_ids": tuple(first["layer_ids"].tolist())}

    # A single list is the generator's shard source. File versions travel with
    # their pair, avoiding ambiguous differently sized list-valued kwargs.
    shards = []
    for feat_path in feat_paths:
        meta_path = feat_path.with_name(feat_path.name.replace("feat.", "meta.")).with_suffix(".jsonl")
        shards.append({
            "features": str(feat_path), "metadata": str(meta_path),
            "versions": [(p.stat().st_size, p.stat().st_mtime_ns) for p in (feat_path, meta_path)],
        })
    return Dataset.from_generator(
        _iter_activation_rows,
        gen_kwargs={"shards": shards, "layout": layout},
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def _metadata_rows(dataset: Dataset) -> list[dict]:
    columns = [key for key in dataset.column_names if key not in ("activations", "layer_ids")]
    return dataset.select_columns(columns).to_list()


def create_manifest_from_activations(dataset: Dataset, **options) -> dict:
    """Create the shared manifest from an Arrow metadata view; no activations are copied."""
    return create_manifest_from_metadata(_metadata_rows(dataset), **options)


def apply_activation_manifest(dataset: Dataset, manifest: dict, *, task_catalog=None,
                              namespace=None, provenance=None, group_column=None,
                              allow_regroup: bool = False) -> Dataset:
    """Annotate an incomplete collection without creating any new assignments."""
    annotations = annotate_activation_metadata(
        _metadata_rows(dataset), manifest, task_catalog=task_catalog, namespace=namespace,
        provenance=provenance, group_column=group_column, allow_regroup=allow_regroup,
    )
    result = dataset
    for key in ("task_uid", "namespace", "source_task_id", "group_id", "grouping", "source_split", "source_provenance", "probe_split", "manifest_id"):
        if key in result.column_names:
            result = result.remove_columns(key)
        values = [annotation[key] for annotation in annotations]
        if key == "source_provenance":
            values = [json.dumps(value, sort_keys=True) for value in values]
        result = result.add_column(key, values)
    return result


def split_activation_dataset(dataset: Dataset, manifest: dict, **identity_options) -> DatasetDict:
    """Return every assigned probe partition, including empty incomplete partitions."""
    annotated = apply_activation_manifest(dataset, manifest, **identity_options)
    names = sorted({entry["probe_split"] for entry in manifest["tasks"].values()
                    if entry["probe_split"] is not None})
    assignments = annotated["probe_split"]
    return DatasetDict({name: annotated.select([i for i, split in enumerate(assignments) if split == name])
                        for name in names})


def load_train_test_split(
    repo_id: str,
    path_in_repo: str = "",
    manifest: Optional[dict | str | Path] = None,
    revision: Optional[str] = None,
    token: Optional[str | bool] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    **identity_options,
) -> DatasetDict:
    """
    Downloads exported activation shards and applies an existing probe manifest.
    path_in_repo selects one run's directory within the repo. Supports private
    repos through token or cached Hugging Face credentials, pinned revisions,
    and offline reuse with local_files_only=True.
    """
    if manifest is None:
        raise ValueError("An existing split manifest is required; collections must not be resplit")
    if isinstance(manifest, (str, Path)):
        manifest = load_split_manifest(manifest)
    subdirectory = PurePosixPath(path_in_repo)
    if subdirectory.is_absolute() or ".." in subdirectory.parts or "\\" in path_in_repo:
        raise ValueError("path_in_repo must be a relative Hugging Face repository directory")
    prefix = f"{subdirectory}/" if str(subdirectory) != "." else ""
    patterns = [
        f"{prefix}{folder}{pattern}"
        for folder in ("", "features/")
        for pattern in ("feat.shard*.npz", "meta.shard*.jsonl")
    ]
    snapshot_path = download_activations(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=patterns,
    )
    dataset = load_activation_dataset(Path(snapshot_path) / str(subdirectory), cache_dir=cache_dir)
    return split_activation_dataset(dataset, manifest, **identity_options)
