"""Persistent probe assignments, independent of official source partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Optional


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_revision(*paths: str | Path) -> str:
    """Fingerprint local generator code or data without inventing a Hub revision."""
    fingerprints = []
    for root_idx, root in enumerate(map(Path, paths)):
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as fp:
                for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                    digest.update(chunk)
            name = str(path.relative_to(root)) if root.is_dir() else path.name
            fingerprints.append((root_idx, name, digest.hexdigest()))
    return content_hash(fingerprints)


def namespaced_id(namespace: str, identifier: str) -> str:
    if not isinstance(namespace, str) or not namespace or not isinstance(identifier, str) or not identifier:
        raise ValueError("Namespace and identifier must be nonempty strings")
    return json.dumps([namespace, identifier], separators=(",", ":"))


@dataclass
class TaskRecord:
    namespace: str
    task_id: str
    source_split: Optional[str] = None
    provenance: dict = field(default_factory=dict)
    group_key: Optional[str] = None
    source_task_id: Optional[str] = None

    @property
    def task_uid(self) -> str:
        return namespaced_id(self.namespace, self.source_task_id if self.source_task_id is not None else self.task_id)

    @property
    def group_id(self) -> str:
        identifier = self.source_task_id if self.source_task_id is not None else self.task_id
        return namespaced_id(self.namespace, self.group_key if self.group_key is not None else identifier)


def _catalog_entries(records: list[TaskRecord]) -> dict:
    entries = {}
    for record in records:
        if record.source_split is not None and (not isinstance(record.source_split, str) or not record.source_split):
            raise ValueError("source_split must be an original partition name or None")
        if record.provenance.get("kind") not in ("huggingface", "generated", "legacy"):
            raise ValueError("Task provenance must declare kind (huggingface, generated, or legacy)")
        if record.provenance["kind"] == "huggingface" and not all(
            record.provenance.get(key) for key in ("dataset_id", "revision")
        ):
            raise ValueError("Hugging Face provenance requires dataset_id and a resolved revision")
        if record.provenance["kind"] == "generated" and not all(
            record.provenance.get(key) for key in ("generator", "generator_revision")
        ):
            raise ValueError("Generated provenance requires a generator and generator_revision")
        if record.provenance["kind"] == "generated" and record.source_split is not None:
            raise ValueError("Generated tasks cannot claim an official source split")
        entry = {
            "namespace": record.namespace,
            "task_id": record.task_id,
            "source_task_id": record.source_task_id if record.source_task_id is not None else record.task_id,
            "group_id": record.group_id,
            "provenance": record.provenance,
            "source_splits": [record.source_split],
        }
        previous = entries.get(record.task_uid)
        if previous is not None:
            for key in ("group_id", "provenance", "task_id"):
                if previous[key] != entry[key]:
                    raise ValueError(f"Conflicting {key} for task {record.task_uid}")
            previous["source_splits"] = sorted(
                set(previous["source_splits"] + entry["source_splits"]), key=lambda value: value or ""
            )
        else:
            entries[record.task_uid] = entry
    if not entries:
        raise ValueError("Cannot create a manifest from an empty task catalog")
    return entries


def create_split_manifest(
    records: list[TaskRecord],
    *,
    policy: str,
    grouping: str = "task",
    test_size: float = 0.2,
    seed: int = 42,
    source_mapping: Optional[dict[str, str]] = None,
    source_splits: Optional[list[str]] = None,
) -> dict:
    """
    Assign a complete intended catalog once, before model-specific collection.
    group_holdout samples groups from the selected source pool; source_mapping
    explicitly maps existing official partitions to probe names. Unselected
    tasks remain in the manifest with probe_split=None.
    """
    entries = _catalog_entries(records)
    available = {split for entry in entries.values() for split in entry["source_splits"]}
    group_assignments = {}
    if policy == "source_mapping":
        if not source_mapping or source_splits is not None:
            raise ValueError("source_mapping policy requires a mapping and no source_splits filter")
        if any(not isinstance(key, str) or not isinstance(value, str) or not key or not value
               for key, value in source_mapping.items()):
            raise ValueError("Source and probe partition names must be nonempty strings")
        missing = set(source_mapping) - available
        if missing:
            raise ValueError(f"Requested official partitions do not exist: {sorted(missing)}")
        for entry in entries.values():
            for source in entry["source_splits"]:
                if source not in source_mapping:
                    continue
                assignment = source_mapping[source]
                previous = group_assignments.setdefault(entry["group_id"], assignment)
                if previous != assignment:
                    raise ValueError(f"Conflicting probe assignments for group {entry['group_id']}")
        settings = {"source_mapping": dict(source_mapping)}
    elif policy == "group_holdout":
        if source_mapping is not None:
            raise ValueError("group_holdout does not accept source_mapping")
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError("test_size must be a fraction strictly between 0 and 1")
        if source_splits is not None and (not source_splits or set(source_splits) - available):
            raise ValueError("Requested source pool contains missing official partitions")
        groups = sorted({entry["group_id"] for entry in entries.values()
                         if source_splits is None or set(entry["source_splits"]).intersection(source_splits)})
        n_test = math.ceil(len(groups) * test_size)
        if not 0 < n_test < len(groups):
            raise ValueError("Group holdout requires nonempty train and test groups")
        random.Random(seed).shuffle(groups)
        group_assignments = {group: "test" if i < n_test else "train" for i, group in enumerate(groups)}
        settings = {"source_splits": source_splits, "test_size": test_size, "seed": seed}
    else:
        raise ValueError("policy must be group_holdout or source_mapping")

    for entry in entries.values():
        # Excluded source rows are not implicitly recruited by a related group.
        selected = (any(split in source_mapping for split in entry["source_splits"])
                    if policy == "source_mapping" else
                    source_splits is None or bool(set(entry["source_splits"]).intersection(source_splits)))
        entry["probe_split"] = group_assignments.get(entry["group_id"]) if selected else None
    manifest = {"version": 1, "policy": policy, "grouping": grouping, "settings": settings,
                "tasks": dict(sorted(entries.items()))}
    manifest["manifest_id"] = content_hash(manifest)
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("version") != 1 or manifest.get("manifest_id") != content_hash(payload):
        raise ValueError("Invalid split manifest version or content hash")
    groups = {}
    for uid, entry in manifest["tasks"].items():
        if uid != namespaced_id(entry["namespace"], entry["source_task_id"]):
            raise ValueError("Manifest task identity does not match its namespace")
        split = entry["probe_split"]
        if split is not None:
            previous = groups.setdefault(entry["group_id"], split)
            if previous != split:
                raise ValueError(f"Conflicting probe assignments for group {entry['group_id']}")


def save_split_manifest(manifest: dict, path: str | Path) -> Path:
    """Never replace an existing assignment file, even with a new seed."""
    validate_split_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return path


def load_split_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def apply_split_manifest(records: list[TaskRecord], manifest: dict) -> list[dict]:
    """Apply saved assignments to any subset; reject unknown tasks or changed sources."""
    validate_split_manifest(manifest)
    annotations = []
    for record in records:
        entry = manifest["tasks"].get(record.task_uid)
        if entry is None:
            raise ValueError(f"Unknown task in split manifest: {record.task_uid}")
        if (entry["group_id"] != record.group_id or entry["provenance"] != record.provenance
                or record.source_split not in entry["source_splits"] or entry["task_id"] != record.task_id):
            raise ValueError(f"Grouping or source provenance differs for task {record.task_uid}")
        selected = (record.source_split in manifest["settings"]["source_mapping"]
                    if manifest["policy"] == "source_mapping" else
                    manifest["settings"]["source_splits"] is None or
                    record.source_split in manifest["settings"]["source_splits"])
        annotations.append({
            "task_uid": record.task_uid, "namespace": record.namespace, "group_id": record.group_id,
            "source_task_id": entry["source_task_id"],
            "source_split": record.source_split, "source_provenance": record.provenance,
            "probe_split": entry["probe_split"] if selected else None, "manifest_id": manifest["manifest_id"],
        })
    return annotations


def activation_records(rows, *, task_catalog=None, namespace=None, provenance=None,
                        grouping="task", group_column=None, allow_regroup=False):
    """Resolve legacy rows explicitly; never infer official provenance from task names."""
    if allow_regroup:
        for column in ("probe_split", "manifest_id"):
            if any(row.get(column) is not None for row in rows):
                raise ValueError("Cannot regroup a collection with existing probe assignments or a manifest")
        if grouping != "task" and task_catalog is None and group_column is None:
            raise ValueError("Explicit regrouping requires a task_catalog or group_column")
    by_id = {}
    for record in task_catalog or []:
        by_id.setdefault(record.task_id, []).append(record)
    records = []
    for original in rows:
        row = dict(original)
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


def create_manifest_from_metadata(rows: list[dict], *, policy: str, task_catalog=None,
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
    observed = activation_records(rows, task_catalog=task_catalog, namespace=namespace,
                                   provenance=provenance, grouping=grouping, group_column=group_column,
                                   allow_regroup=allow_regroup)
    manifest = create_split_manifest(task_catalog if task_catalog is not None else observed,
                                     policy=policy, grouping=grouping, **policy_options)
    _check_existing_annotations(rows, apply_split_manifest(observed, manifest))
    return manifest


def _check_existing_annotations(rows: list[dict], annotations: list[dict]) -> None:
    for row, annotation in zip(rows, annotations):
        for key in ("probe_split", "manifest_id"):
            if row.get(key) is not None and row[key] != annotation[key]:
                raise ValueError(f"Existing {key} conflicts with saved manifest")


def annotate_activation_metadata(rows: list[dict], manifest: dict, **identity_options) -> list[dict]:
    records = activation_records(rows, grouping=manifest["grouping"], **identity_options)
    annotations = apply_split_manifest(records, manifest)
    _check_existing_annotations(rows, annotations)
    return [dict(row, **annotation, grouping=manifest["grouping"])
            for row, annotation in zip(rows, annotations)]
