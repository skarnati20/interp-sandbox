"""Tests for loading activation shards and preparing Hugging Face splits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_from_disk
from datasets.exceptions import DatasetGenerationError
import numpy as np

from src.data import (create_manifest_from_activations, split_activation_dataset,
                      apply_activation_manifest, load_activation_dataset, load_train_test_split)
from src.splits import TaskRecord, create_split_manifest


class TestActivationData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.run_dir = self.directory / "benchmark" / "model"
        self.features_dir = self.run_dir / "features"
        self.features_dir.mkdir(parents=True)
        self.cache_dir = self.directory / "cache"

        # Each task spans rollouts, steps, anchors, and multiple feature shards.
        self.records = [
            {
                "task_id": f"textcraft/task_{task}",
                "episode_id": f"task_{task}_k{rollout}",
                "rollout_k": rollout,
                "step_idx": step,
                "anchor": anchor,
                "success": task % 2,
            }
            for task in range(4)
            for rollout in range(2)
            for step in range(2)
            for anchor in ("post_gen", "pre_gen")
        ]
        self.features = np.arange(len(self.records) * 6, dtype=np.float16).reshape(-1, 2, 3)
        self._write_shard(2, self.features[:17], self.records[:17])
        self._write_shard(10, self.features[17:], self.records[17:])

    def _catalog(self):
        return [TaskRecord("fixture", f"textcraft/task_{i}", provenance={"kind": "legacy", "collection": "fixture"})
                for i in range(4)]

    def _manifest(self, seed=7):
        return create_split_manifest(self._catalog(), policy="group_holdout", test_size=0.25, seed=seed)

    def _split(self, dataset, seed=7):
        return split_activation_dataset(dataset, self._manifest(seed), task_catalog=self._catalog())

    def _write_shard(self, shard_idx, features, records, layer_ids=(4, 10)):
        np.savez_compressed(
            self.features_dir / f"feat.shard{shard_idx}.npz",
            X=features,
            layer_ids=np.array(layer_ids),
        )
        metadata = "".join(json.dumps(record) + "\n" for record in records)
        (self.features_dir / f"meta.shard{shard_idx}.jsonl").write_text(metadata, encoding="utf-8")

    def _load(self):
        return load_activation_dataset(self.run_dir, cache_dir=self.cache_dir)

    def test_load_preserves_features_and_metadata_in_numeric_shard_order(self):
        dataset = self._load()
        self.assertEqual(len(dataset), len(self.records))
        self.assertEqual(dataset[17]["task_id"], self.records[17]["task_id"])
        self.assertEqual(dataset[17]["anchor"], self.records[17]["anchor"])
        self.assertEqual(dataset[17]["success"], self.records[17]["success"])
        self.assertEqual(dataset[0]["layer_ids"], [4, 10])
        self.assertEqual(dataset.features["activations"].feature.feature.dtype, "float16")
        np.testing.assert_array_equal(dataset["activations"], self.features)
        direct = load_activation_dataset(self.features_dir, cache_dir=self.cache_dir)
        self.assertEqual(direct[:], dataset[:])

    def test_split_keeps_tasks_together_and_is_reproducible(self):
        dataset = self._load()
        splits = self._split(dataset)
        self.assertIsInstance(splits, DatasetDict)
        self.assertEqual(len(splits["train"]), 24)
        self.assertEqual(len(splits["test"]), 8)
        train_tasks = set(splits["train"]["task_id"])
        test_tasks = set(splits["test"]["task_id"])
        self.assertTrue(train_tasks.isdisjoint(test_tasks))
        self.assertEqual(train_tasks | test_tasks, set(dataset["task_id"]))
        self.assertTrue(set(splits["train"]["episode_id"]).isdisjoint(splits["test"]["episode_id"]))
        again = self._split(dataset)
        self.assertEqual(splits["test"][:], again["test"][:])
        reversed_dataset = dataset.select(range(len(dataset) - 1, -1, -1))
        reversed_splits = self._split(reversed_dataset)
        self.assertEqual(test_tasks, set(reversed_splits["test"]["task_id"]))

    def test_bootstrap_from_existing_incomplete_collection_uses_full_catalog(self):
        dataset = self._load()
        subset = dataset.select([0, 1, 2])
        manifest = create_manifest_from_activations(
            subset, policy="group_holdout", task_catalog=self._catalog(), test_size=0.25, seed=7)
        self.assertEqual(manifest, self._manifest())
        incomplete = split_activation_dataset(subset, manifest, task_catalog=self._catalog())
        self.assertEqual(sum(len(partition) for partition in incomplete.values()), 3)
        self.assertTrue(any(len(partition) == 0 for partition in incomplete.values()))
        full = split_activation_dataset(dataset, manifest, task_catalog=self._catalog())
        self.assertEqual(len(full["test"]), 8)

    def test_legacy_requires_provenance_and_unknown_tasks_are_rejected(self):
        dataset = self._load()
        with self.assertRaisesRegex(ValueError, "explicit namespace"):
            create_manifest_from_activations(dataset, policy="group_holdout")
        manifest = create_manifest_from_activations(
            dataset, policy="group_holdout", namespace="fixture",
            provenance={"kind": "legacy", "collection": "fixture"}, test_size=0.25, seed=7)
        self.assertEqual(manifest, self._manifest())
        unknown = Dataset.from_dict({"task_id": ["unknown"]})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            split_activation_dataset(unknown, manifest, task_catalog=self._catalog())

    def test_existing_probe_assignments_cannot_be_overwritten(self):
        annotated = apply_activation_manifest(self._load(), self._manifest(), task_catalog=self._catalog())
        wrong = annotated.remove_columns("probe_split").add_column("probe_split", ["wrong"] * len(annotated))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            split_activation_dataset(wrong, self._manifest(), task_catalog=self._catalog())
        with self.assertRaisesRegex(ValueError, "conflicts"):
            create_manifest_from_activations(wrong, policy="group_holdout", task_catalog=self._catalog(),
                                             test_size=0.25, seed=7)

    def test_legacy_goal_groups_and_ambiguous_namespaces(self):
        provenance = {"kind": "legacy", "collection": "goals"}
        rows = Dataset.from_dict({"task_id": ["a", "b", "c"], "goal": ["oak_boat", "oak_boat", "sticks"]})
        manifest = create_manifest_from_activations(
            rows, policy="group_holdout", namespace="textcraft", provenance=provenance,
            grouping="crafting_goal", group_column="goal", test_size=0.5)
        annotated = apply_activation_manifest(rows, manifest, namespace="textcraft",
                                              provenance=provenance, group_column="goal")
        self.assertEqual(annotated[0]["probe_split"], annotated[1]["probe_split"])
        self.assertNotEqual(annotated[0]["probe_split"], annotated[2]["probe_split"])
        ambiguous = [TaskRecord("one", "a", provenance=provenance), TaskRecord("two", "a", provenance=provenance)]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            create_manifest_from_activations(rows.select([0]), policy="group_holdout", task_catalog=ambiguous)

    def test_duplicate_display_ids_resolve_by_original_source_identity(self):
        provenance = {"kind": "legacy", "collection": "fixture"}
        catalog = [TaskRecord("fixture", "0", provenance=provenance, source_task_id=source)
                   for source in ("source-a/0", "source-b/0")]
        rows = Dataset.from_dict({"task_id": ["0", "0"], "source_task_id": ["source-a/0", "source-b/0"]})
        manifest = create_manifest_from_activations(rows, policy="group_holdout", task_catalog=catalog, test_size=0.5)
        splits = split_activation_dataset(rows, manifest, task_catalog=catalog)
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["test"]), 1)

    def test_empty_collection_retains_saved_probe_partitions(self):
        empty = self._load().select([])
        splits = split_activation_dataset(empty, self._manifest(), task_catalog=self._catalog())
        self.assertEqual(set(splits), {"train", "test"})
        self.assertEqual([len(partition) for partition in splits.values()], [0, 0])

    def test_splits_can_be_saved_and_reloaded_as_hugging_face_datasets(self):
        splits = self._split(self._load())
        output_dir = self.directory / "splits"
        splits.save_to_disk(str(output_dir))
        restored = load_from_disk(str(output_dir))
        self.assertEqual(restored["train"][:], splits["train"][:])
        self.assertEqual(restored["test"][:], splits["test"][:])

    def test_empty_directory_and_missing_metadata_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_activation_dataset(self.directory / "missing", cache_dir=self.cache_dir)
        (self.features_dir / "meta.shard10.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "matching metadata"):
            self._load()

    def test_row_misalignment_is_rejected(self):
        self._write_shard(10, self.features[17:], self.records[18:])
        with self.assertRaises(DatasetGenerationError) as raised:
            self._load()
        self.assertIn("row count mismatch", str(raised.exception.__cause__))

    def test_layer_misalignment_is_rejected(self):
        self._write_shard(10, self.features[17:], self.records[17:], layer_ids=(10, 4))
        with self.assertRaises(DatasetGenerationError) as raised:
            self._load()
        self.assertIn("layer_ids differ", str(raised.exception.__cause__))

    def test_replaced_shards_invalidate_the_dataset_cache(self):
        original = self._load()
        changed_records = [dict(record, success=1) for record in self.records[:17]]
        self._write_shard(2, self.features[:17], changed_records)
        changed = self._load()
        self.assertEqual(original[0]["success"], 0)
        self.assertEqual(changed[0]["success"], 1)

    @patch("src.hf.snapshot_download")
    def test_hub_download_selects_one_run_and_passes_download_options(self, download):
        download.return_value = str(self.directory)
        splits = load_train_test_split(
            "owner/activations",
            path_in_repo="benchmark/model",
            manifest=self._manifest(),
            task_catalog=self._catalog(),
            revision="fixed-revision",
            token="test-token",
            cache_dir=self.cache_dir,
            local_files_only=True,
        )
        self.assertEqual(len(splits["test"]), 8)
        download.assert_called_once_with(
            repo_id="owner/activations",
            repo_type="dataset",
            revision="fixed-revision",
            token="test-token",
            cache_dir=self.cache_dir,
            local_files_only=True,
            local_dir=None,
            allow_patterns=[
                "benchmark/model/feat.shard*.npz",
                "benchmark/model/meta.shard*.jsonl",
                "benchmark/model/features/feat.shard*.npz",
                "benchmark/model/features/meta.shard*.jsonl",
            ],
        )

    @patch("src.hf.snapshot_download")
    def test_invalid_hub_directory_is_rejected_before_download(self, download):
        for directory in ("../other", "/absolute", "run/../../other"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                load_train_test_split("owner/activations", path_in_repo=directory, manifest=self._manifest())
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
