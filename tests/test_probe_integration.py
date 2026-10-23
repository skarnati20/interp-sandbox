"""Compatibility with the owner's ActivationDataset, split controls, and probes."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import pandas as pd

from src.data import split_activation_dataset as huggingface_splits
from src.probe.base import Probe
from src.probe.probes.linear_probe import LinearProbe
from src.probe.probes.mlp_probe import MLPProbe
from src.probe.probes.xgboost_probe import XGBoostProbe
from src.probe.split import (apply_activation_manifest, create_manifest_from_activations,
                             split_activation_dataset)
from src.probe.train import train_and_evaluate
from src.splits import TaskRecord, create_split_manifest, namespaced_id, save_split_manifest
from src.store import ActivationDataset, ShardReader, ShardWriter


class TestProbeIntegration(unittest.TestCase):
    def setUp(self):
        self.catalog = [TaskRecord("fixture", str(i), provenance={"kind": "legacy", "collection": "fixture"})
                        for i in range(8)]
        rows = [{"task_id": str(i), "success": outcome, "round": turn, "row_id": f"{i}-{outcome}-{turn}"}
                for i in range(8) for outcome in (0, 1) for turn in (0, 1)]
        labels = np.array([row["success"] for row in rows])
        self.dataset = ActivationDataset(
            np.column_stack([labels * 2 - 1, np.arange(len(rows)) / len(rows)]).astype(np.float32),
            labels, [3], pd.DataFrame(rows))
        self.manifest = create_split_manifest(self.catalog, policy="group_holdout", test_size=0.5, seed=7)
        self.options = {"manifest": self.manifest, "task_catalog": self.catalog}

    def test_manifest_matches_huggingface_view_without_copying_features(self):
        annotated = apply_activation_manifest(self.dataset, self.manifest, task_catalog=self.catalog)
        self.assertIs(annotated.X, self.dataset.X)
        self.assertNotIn("probe_split", self.dataset.metadata)
        native = split_activation_dataset(self.dataset, **self.options)
        arrow = huggingface_splits(Dataset.from_pandas(self.dataset.metadata), self.manifest,
                                   task_catalog=self.catalog)
        for actual, name in zip(native, ("train", "test")):
            self.assertEqual(actual.metadata["row_id"].tolist(), arrow[name]["row_id"])
            self.assertEqual(set(actual.metadata["manifest_id"]), {self.manifest["manifest_id"]})
            np.testing.assert_array_equal(actual.y, actual.metadata["success"])

    def test_incomplete_models_keep_assignments_and_empty_partitions(self):
        train, test = split_activation_dataset(self.dataset, **self.options)
        for full, absent_name in ((train, "test"), (test, "train")):
            incomplete = full.slice(np.array([0, 1]))
            first, second = split_activation_dataset(incomplete, **self.options)
            self.assertEqual((first.num_samples, second.num_samples),
                             (2, 0) if absent_name == "test" else (0, 2))
            self.assertEqual(first.X.shape[1:], self.dataset.X.shape[1:])
            self.assertEqual(second.layer_ids, [3])
        empty = self.dataset.slice(np.array([], dtype=int))
        self.assertEqual([part.num_samples for part in split_activation_dataset(empty, **self.options)], [0, 0])

    def test_manifest_ignores_exploratory_split_size_and_seed(self):
        original = split_activation_dataset(self.dataset, **self.options)
        changed = split_activation_dataset(self.dataset, eval_size=0.1, seed=999, group_by="round", **self.options)
        self.assertEqual([part.metadata["row_id"].tolist() for part in original],
                         [part.metadata["row_id"].tolist() for part in changed])

    def test_native_collect_first_bootstrap_uses_full_catalog(self):
        incomplete = self.dataset.slice(np.array([0, 1]))
        manifest = create_manifest_from_activations(incomplete, policy="group_holdout", task_catalog=self.catalog,
                                                   test_size=0.5, seed=7)
        self.assertEqual(manifest, self.manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = save_split_manifest(manifest, Path(directory) / "splits.json")
            first, second = split_activation_dataset(self.dataset, manifest=path, task_catalog=self.catalog)
        self.assertEqual((first.num_samples, second.num_samples), (16, 16))

    def test_native_collect_first_regrouping_and_conflicts(self):
        goals = [replace(record, group_key=f"goal-{i // 2}") for i, record in enumerate(self.catalog)]
        provisional = self.dataset.metadata.copy()
        provisional["namespace"] = "fixture"
        provisional["grouping"] = "task"
        provisional["group_id"] = [namespaced_id("fixture", task) for task in provisional["task_id"]]
        provisional["probe_split"] = None
        provisional["manifest_id"] = None
        collected = replace(self.dataset, metadata=provisional)
        with self.assertRaisesRegex(ValueError, "Group differs"):
            create_manifest_from_activations(collected, policy="group_holdout", grouping="crafting_goal",
                                             task_catalog=goals)
        manifest = create_manifest_from_activations(collected, policy="group_holdout", grouping="crafting_goal",
                                                    task_catalog=goals, allow_regroup=True)
        annotated = apply_activation_manifest(collected, manifest, task_catalog=goals, allow_regroup=True)
        self.assertEqual(annotated.metadata.groupby("group_id")["probe_split"].nunique().max(), 1)
        with self.assertRaisesRegex(ValueError, "Cannot regroup"):
            apply_activation_manifest(annotated, manifest, task_catalog=goals, allow_regroup=True)
        wrong = annotated.metadata.copy()
        wrong.loc[0, "probe_split"] = "wrong"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            split_activation_dataset(replace(annotated, metadata=wrong), manifest=manifest, task_catalog=goals)

    def test_namespaced_tasks_do_not_collide_and_changed_provenance_is_rejected(self):
        provenance = {"kind": "generated", "generator": "fixture", "generator_revision": "revision-1"}
        catalog = [TaskRecord(namespace, "same-id", provenance=provenance) for namespace in ("one", "two")]
        manifest = create_split_manifest(catalog, policy="group_holdout", test_size=0.5)
        dataset = self.dataset.slice(np.array([0, 1]))
        metadata = pd.DataFrame([{"task_id": "same-id", "namespace": namespace} for namespace in ("one", "two")])
        dataset = replace(dataset, metadata=metadata)
        train, test = split_activation_dataset(dataset, manifest=manifest, task_catalog=catalog)
        self.assertEqual((train.num_samples, test.num_samples), (1, 1))
        annotated = apply_activation_manifest(dataset, manifest, task_catalog=catalog)
        changed = [replace(record, provenance={**provenance, "generator_revision": "revision-2"})
                   for record in catalog]
        with self.assertRaisesRegex(ValueError, "Source provenance differs"):
            split_activation_dataset(annotated, manifest=manifest, task_catalog=changed)

    def test_assigned_collection_cannot_use_random_split(self):
        annotated = apply_activation_manifest(self.dataset, self.manifest, task_catalog=self.catalog)
        with self.assertRaisesRegex(ValueError, "saved manifest"):
            split_activation_dataset(annotated)
        # Null annotations are provisional and retain the owner's original API.
        provisional = self.dataset.metadata.assign(probe_split=None, manifest_id=None)
        train, test = split_activation_dataset(replace(self.dataset, metadata=provisional), eval_size=0.25)
        self.assertEqual((train.num_samples, test.num_samples), (24, 8))
        self.assertTrue(set(train.metadata.task_id).isdisjoint(test.metadata.task_id))

    def test_divergence_and_rebalancing_preserve_manifest_membership(self):
        before_train, before_test = split_activation_dataset(self.dataset, **self.options)
        train, test = split_activation_dataset(self.dataset, divergent_tasks_only=True,
                                               target_success_ratio=0.5, **self.options)
        self.assertEqual(set(train.metadata.task_uid), set(before_train.metadata.task_uid))
        self.assertEqual(test.metadata.row_id.tolist(), before_test.metadata.row_id.tolist())
        self.assertEqual(train.y.mean(), 0.5)
        # Filtering one model's convergent task cannot send another task across partitions.
        changed = self.dataset.metadata.copy()
        changed.loc[changed.task_id == "0", "success"] = 0
        changed_dataset = replace(self.dataset, metadata=changed, y=changed.success.to_numpy())
        filtered = split_activation_dataset(changed_dataset, divergent_tasks_only=True, **self.options)
        self.assertTrue(all("0" not in set(part.metadata.task_id) for part in filtered))
        for actual, original in zip(filtered, (before_train, before_test)):
            self.assertTrue(set(actual.metadata.task_uid) <= set(original.metadata.task_uid))

    def test_explicit_partition_names_and_missing_official_train(self):
        catalog = [replace(record, source_split="test") for record in self.catalog]
        test_only = create_split_manifest(catalog, policy="source_mapping", source_mapping={"test": "test"})
        with self.assertRaisesRegex(ValueError, "distinct requested"):
            split_activation_dataset(self.dataset, manifest=test_only, task_catalog=catalog)
        manifest = create_split_manifest(catalog, policy="group_holdout", test_size=0.5)
        # Legacy rows use the authoritative catalog and keep official test as provenance.
        train, test = split_activation_dataset(self.dataset, manifest=manifest, task_catalog=catalog)
        self.assertEqual(set(train.metadata.source_split) | set(test.metadata.source_split), {"test"})
        mapped = create_split_manifest([replace(record, source_split="a" if i < 4 else "b")
                                        for i, record in enumerate(self.catalog)], policy="source_mapping",
                                       source_mapping={"a": "fit", "b": "validation"})
        mapped_catalog = [replace(record, source_split="a" if i < 4 else "b")
                          for i, record in enumerate(self.catalog)]
        train, test = split_activation_dataset(self.dataset, manifest=mapped, task_catalog=mapped_catalog,
                                               train_split="fit", eval_split="validation")
        self.assertEqual((train.num_samples, test.num_samples), (16, 16))

    def test_training_pipeline_uses_owner_probes_and_separate_turn_cutoffs(self):
        probes = [LinearProbe(), MLPProbe(hidden_layer_sizes=(4,), early_stopping=False, max_iter=1000),
                  XGBoostProbe(n_estimators=4, max_depth=2)]
        for probe in probes:
            with self.subTest(probe=type(probe).__name__):
                self.assertIsInstance(probe, Probe)
                result = train_and_evaluate(probe, dataset=self.dataset, train_max_turn=1,
                                            eval_max_turn=2, **self.options)
                self.assertEqual(result["train_samples"], 8)
                self.assertEqual(result["eval_samples"], 16)
                self.assertTrue(np.isfinite(result["eval_metrics"]["accuracy"]))
        train, test = split_activation_dataset(self.dataset, **self.options)
        result = train_and_evaluate(LinearProbe(), train_set=train, eval_set=test, max_turn=1)
        self.assertEqual(result["eval_samples"], 8)
        with self.assertRaisesRegex(ValueError, "pre-split"):
            train_and_evaluate(LinearProbe(), train_set=train, eval_set=test, manifest=self.manifest)

    def test_owner_reader_and_huggingface_view_share_validation(self):
        from src.data import load_activation_dataset
        with tempfile.TemporaryDirectory() as directory:
            writer = ShardWriter(Path(directory) / "features", shard_size=3, layer_ids=[3])
            for feature, row in zip(self.dataset.X[:6], self.dataset.metadata.to_dict("records")[:6]):
                writer.add_step(feature, row)
            writer.close()
            reader = ShardReader(directory)
            native = reader.read_all(anchor=None, layer=3)
            arrow = load_activation_dataset(directory, cache_dir=Path(directory) / "cache")
            np.testing.assert_array_equal(native.X, np.array(arrow["activations"])[:, 0, :])
            self.assertEqual(native.metadata.row_id.tolist(), arrow["row_id"])
            np.savez_compressed(Path(directory) / "features/feat.shard1.npz", X=np.zeros((3, 1, 2)), layer_ids=[4])
            with self.assertRaisesRegex(ValueError, "layer_ids differ"):
                reader.read_all()
            (Path(directory) / "features/meta.shard0.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "matching metadata"):
                reader.read_all()

    @patch("src.hf.snapshot_download")
    def test_owner_hub_reader_passes_revision_and_offline_options(self, download):
        with tempfile.TemporaryDirectory() as directory:
            download.return_value = directory
            dataset = ShardReader.read_from_huggingface(
                "owner/activations", revision="fixed-sha", local_files_only=True, cache_dir=directory)
            self.assertEqual(dataset.num_samples, 0)
            self.assertEqual(download.call_args.kwargs["revision"], "fixed-sha")
            self.assertTrue(download.call_args.kwargs["local_files_only"])
