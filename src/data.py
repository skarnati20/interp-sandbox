"""Load collected activations and prepare Hugging Face train/test datasets."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
import numpy as np

from src.splits import TaskRecord, apply_split_manifest, create_split_manifest, load_split_manifest


def _iter_activation_rows(shards: list[dict], layout: dict) -> Iterator[dict]:
    expected_shape = tuple(layout["shape"])
    expected_layers = layout["layer_ids"]

    for descriptor in shards:
        feat_path = Path(descriptor["features"])
        meta_path = Path(descriptor["metadata"])

        with np.load(feat_path, allow_pickle=False) as shard:
            if "X" not in shard or "layer_ids" not in shard:
                raise ValueError(f"Missing X or layer_ids in {feat_path}")
            features = shard["X"]
            layer_ids = shard["layer_ids"]
            if features.ndim != 3 or layer_ids.ndim != 1 or len(layer_ids) != features.shape[1]:
                raise ValueError(f"Expected X shaped [rows, layers, hidden_dim] and matching layer_ids in {feat_path}")
            if features.shape[1:] != expected_shape or not np.array_equal(layer_ids, expected_layers):
                raise ValueError(f"Activation shape or layer_ids differ across shards: {feat_path}")

            with open(meta_path, "r", encoding="utf-8") as fp:
                metadata = [json.loads(line) for line in fp if line.strip()]
            if len(metadata) != len(features):
                raise ValueError(f"Feature/metadata row count mismatch for {feat_path}")

            for activation, meta in zip(features, metadata):
                if not isinstance(meta, dict) or "activations" in meta:
                    raise ValueError(f"Metadata must be an object without an activations key: {meta_path}")
                if "layer_ids" in meta and meta["layer_ids"] != layer_ids.tolist():
                    raise ValueError(f"Recorded layer_ids differ from feature shard: {meta_path}")
                if "source_provenance" in meta:
                    meta["source_provenance"] = json.dumps(meta["source_provenance"], sort_keys=True)
                yield {
                    **meta,
                    "activations": activation.astype(np.float16, copy=False),
                    "layer_ids": layer_ids.tolist(),
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
    feat_paths = list(features_dir.glob("feat.shard*.npz"))
    if not feat_paths:
        raise FileNotFoundError(f"No activation shards found in {features_dir}")
    feat_paths.sort(key=lambda p: int(p.stem.removeprefix("feat.shard")))

    expected_meta_paths = {
        p.with_name(p.name.replace("feat.", "meta.")).with_suffix(".jsonl") for p in feat_paths
    }
    actual_meta_paths = set(features_dir.glob("meta.shard*.jsonl"))
    if expected_meta_paths != actual_meta_paths:
        raise ValueError(f"Every feature shard must have a matching metadata shard in {features_dir}")

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


def _activation_records(dataset, *, task_catalog=None, namespace=None, provenance=None,
                        grouping="task", group_column=None, allow_regroup=False):
    """Resolve legacy rows explicitly; never infer official provenance from task names."""
    if allow_regroup:
        for column in ("probe_split", "manifest_id"):
            if column in dataset.column_names and any(value is not None for value in dataset[column]):
                raise ValueError("Cannot regroup a collection with existing probe assignments or a manifest")
        if grouping != "task" and task_catalog is None and group_column is None:
            raise ValueError("Explicit regrouping requires a task_catalog or group_column")
    by_id = {}
    for record in task_catalog or []:
        by_id.setdefault(record.task_id, []).append(record)
    columns = [column for column in dataset.column_names if column not in ("activations", "layer_ids")]
    records = []
    for row in dataset.select_columns(columns):
        if isinstance(row.get("source_provenance"), str):
            row["source_provenance"] = json.loads(row["source_provenance"])
        task_id = row.get("task_id")
        row_namespace = row.get("namespace") or namespace
        candidates = [record for record in by_id.get(task_id, [])
                      if row_namespace is None or record.namespace == row_namespace]
        if row.get("source_split") is not None:
            candidates = [record for record in candidates if record.source_split == row["source_split"]]
        if row.get("source_task_id") is not None:
            candidates = [record for record in candidates
                          if (record.source_task_id or record.task_id) == row["source_task_id"]]
        if row.get("task_uid") is not None:
            candidates = [record for record in candidates if record.task_uid == row["task_uid"]]
        if task_catalog is not None:
            if len(candidates) != 1:
                raise ValueError(f"Unknown or ambiguous task in supplied catalog: {task_id}")
            record = candidates[0]
            # Validate any existing provenance/identity rather than overwrite it.
            if row.get("source_provenance") is not None and row["source_provenance"] != record.provenance:
                raise ValueError(f"Source provenance differs for task {task_id}")
            if row.get("source_task_id") is not None and row["source_task_id"] != (record.source_task_id or record.task_id):
                raise ValueError(f"Original source task ID differs for task {task_id}")
        else:
            source = row.get("source_provenance") or provenance
            if row_namespace is None or not source:
                raise ValueError("Legacy rows require an explicit namespace and provenance, or a task_catalog")
            if group_column:
                if row.get(group_column) is None or row[group_column] == "":
                    raise ValueError(f"Missing grouping column: {group_column}")
                group_key = str(row[group_column])
            elif grouping == "task":
                group_key = row.get("source_task_id") or task_id
            elif row.get("group_id"):
                saved_namespace, group_key = json.loads(row["group_id"])
                if saved_namespace != row_namespace:
                    raise ValueError("Group namespace differs from task namespace")
            else:
                raise ValueError("Non-task grouping requires a catalog or explicit group_column")
            record = TaskRecord(row_namespace, task_id, row.get("source_split"), source, group_key,
                                source_task_id=row.get("source_task_id"))
        if row.get("task_uid") is not None and row["task_uid"] != record.task_uid:
            raise ValueError(f"Task identity differs for task {task_id}")
        if not allow_regroup and row.get("group_id") is not None and row["group_id"] != record.group_id:
            raise ValueError(f"Group differs for task {task_id}")
        if not allow_regroup and row.get("grouping") is not None and row["grouping"] != grouping:
            raise ValueError(f"Grouping mode differs for task {task_id}")
        records.append(record)
    return records


def create_manifest_from_activations(dataset: Dataset, *, policy: str, task_catalog=None,
                                     namespace=None, provenance=None, grouping="task",
                                     group_column=None, allow_regroup: bool = False, **policy_options) -> dict:
    """
    Bootstrap one manifest for existing collections. With task_catalog, use the
    full supplied universe even if the collection is incomplete. Without it,
    only observed tasks are known; explicitly describe unknown legacy provenance.
    Save the result with save_split_manifest and reuse it for every model.
    allow_regroup explicitly replaces provisional groups only when the entire
    input collection has no probe assignments or manifest IDs.
    """
    observed = _activation_records(dataset, task_catalog=task_catalog, namespace=namespace,
                                   provenance=provenance, grouping=grouping, group_column=group_column,
                                   allow_regroup=allow_regroup)
    manifest = create_split_manifest(task_catalog if task_catalog is not None else observed,
                                     policy=policy, grouping=grouping, **policy_options)
    _check_existing_annotations(dataset, apply_split_manifest(observed, manifest))
    return manifest


def _check_existing_annotations(dataset: Dataset, annotations: list[dict]) -> None:
    for key in ("probe_split", "manifest_id"):
        if key in dataset.column_names:
            for row, annotation in zip(dataset.select_columns([key]), annotations):
                if row[key] is not None and row[key] != annotation[key]:
                    raise ValueError(f"Existing {key} conflicts with saved manifest")


def apply_activation_manifest(dataset: Dataset, manifest: dict, *, task_catalog=None,
                              namespace=None, provenance=None, group_column=None,
                              allow_regroup: bool = False) -> Dataset:
    """Annotate an incomplete collection without creating any new assignments."""
    records = _activation_records(dataset, task_catalog=task_catalog, namespace=namespace,
                                  provenance=provenance, grouping=manifest["grouping"], group_column=group_column,
                                  allow_regroup=allow_regroup)
    annotations = apply_split_manifest(records, manifest)
    _check_existing_annotations(dataset, annotations)
    for annotation in annotations:
        annotation["grouping"] = manifest["grouping"]
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
    snapshot_path = snapshot_download(
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
