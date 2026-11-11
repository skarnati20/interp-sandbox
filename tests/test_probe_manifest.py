"""Saved probe splits stay fixed across models and incomplete collections."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.probe.manifest import create_split_manifest, load_split_manifest, save_split_manifest
from src.probe.probes.linear_probe import LinearProbe
from src.probe.split import split_activation_dataset
from src.probe.train import train_and_evaluate
from src.store import ActivationDataset


def dataset(metadata):
    metadata = metadata.reset_index(drop=True)
    y = metadata["success"].to_numpy()
    return ActivationDataset(np.column_stack([y * 2 - 1, np.arange(len(y))]).astype(np.float32),
                             y, [0], metadata)


class TestProbeManifest(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            {"task_id": f"task-{task}", "success": outcome, "round": turn, "source_split": "test"}
            for task in range(8) for outcome in (0, 1) for turn in (0, 1)
        ])
        self.data = dataset(self.rows)
        self.manifest = create_split_manifest(self.rows, eval_size=0.5)

    def split(self, data=None, **options):
        return split_activation_dataset(self.data if data is None else data, manifest=self.manifest, **options)

    def test_round_trip_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            save_split_manifest(self.manifest, path)
            self.assertEqual(load_split_manifest(path), self.manifest)
            self.assertEqual(json.loads(path.read_text()), self.manifest)
            train, test = split_activation_dataset(self.data, manifest=path)
            self.assertEqual((train.num_samples, test.num_samples), (16, 16))
            with self.assertRaises(FileExistsError):
                save_split_manifest(self.manifest, path)

    def test_creation_ignores_row_order_and_repeated_rollouts(self):
        reordered = self.rows.sample(frac=1, random_state=3)
        self.assertEqual(create_split_manifest(reordered, eval_size=0.5), self.manifest)
        self.assertEqual(create_split_manifest(self.rows.drop_duplicates("task_id"), eval_size=0.5), self.manifest)

    def test_same_assignments_across_models_and_split_options(self):
        train, test = self.split()
        other = dataset(self.rows.sample(frac=1, random_state=12))
        other_train, other_test = self.split(other, seed=99, eval_size=0.1)
        self.assertEqual(set(train.metadata.task_id), set(other_train.metadata.task_id))
        self.assertEqual(set(test.metadata.task_id), set(other_test.metadata.task_id))
        self.assertTrue(set(train.metadata.task_id).isdisjoint(test.metadata.task_id))
        pd.testing.assert_frame_equal(self.data.metadata, self.rows)
        self.assertEqual(set(train.metadata.source_split) | set(test.metadata.source_split), {"test"})

    def test_incomplete_and_empty_collections_keep_saved_partitions(self):
        full_train, full_test = self.split()
        partial_train, partial_test = self.split(full_train.slice(np.array([0, 1])))
        self.assertEqual((partial_train.num_samples, partial_test.num_samples), (2, 0))
        partial_train, partial_test = self.split(full_test.slice(np.array([0, 1])))
        self.assertEqual((partial_train.num_samples, partial_test.num_samples), (0, 2))
        empty = self.data.slice(np.array([], dtype=int))
        self.assertEqual([part.num_samples for part in self.split(empty)], [0, 0])
        self.assertEqual(partial_test.layer_ids, self.data.layer_ids)

    def test_can_create_full_catalog_without_activations(self):
        catalog = pd.DataFrame({"task_id": [f"task-{i}" for i in range(8)]})
        self.assertEqual(create_split_manifest(catalog, eval_size=0.5), self.manifest)
        first_model = dataset(self.rows[self.rows.task_id.isin(["task-0", "task-2"])])
        self.assertEqual(sum(part.num_samples for part in self.split(first_model)), first_model.num_samples)

    def test_unknown_groups_are_rejected_even_if_outcome_filter_would_drop_them(self):
        unknown = dataset(pd.DataFrame([{"task_id": "new-task", "success": 0}]))
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            self.split(unknown, divergent_tasks_only=True)

    def test_conflicting_and_duplicate_assignments_are_rejected(self):
        conflicting = copy.deepcopy(self.manifest)
        conflicting["test"].append(conflicting["train"][0])
        duplicate = copy.deepcopy(self.manifest)
        duplicate["train"].append(duplicate["train"][0])
        for manifest in (conflicting, duplicate):
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "conflicting assignment"):
                split_activation_dataset(self.data, manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.json"
            path.write_text(json.dumps(conflicting))
            with self.assertRaisesRegex(ValueError, "conflicting assignment"):
                load_split_manifest(path)

    def test_composite_keys_namespace_tasks_and_goals(self):
        rows = pd.concat([self.rows.assign(benchmark=name, goal=lambda frame: frame.task_id.str[-1].astype(int) // 2)
                          for name in ("one", "two")], ignore_index=True)
        manifest = create_split_manifest(rows, group_by=["benchmark", "goal"], eval_size=0.5)
        self.assertEqual(sum(len(manifest[split]) for split in ("train", "test")), 8)
        train, test = split_activation_dataset(dataset(rows), manifest=manifest)
        train_groups = set(zip(train.metadata.benchmark, train.metadata.goal))
        test_groups = set(zip(test.metadata.benchmark, test.metadata.goal))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertEqual(len(train_groups | test_groups), 8)

    def test_missing_group_columns_and_values_are_errors(self):
        for rows, columns in ((self.rows, "goal"), (self.rows.assign(task_id=None), "task_id"),
                              (self.rows, []), (self.rows, ["task_id", "task_id"])):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                create_split_manifest(rows, group_by=columns)
        with self.assertRaisesRegex(ValueError, "Missing grouping"):
            self.split(dataset(self.rows.drop(columns="task_id")))
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.split(dataset(self.rows.assign(task_id=None)))

    def test_divergence_and_balancing_do_not_change_assignments(self):
        original_train, original_test = self.split()
        train, test = self.split(divergent_tasks_only=True, target_success_ratio=0.5)
        self.assertEqual(set(train.metadata.task_id), set(original_train.metadata.task_id))
        pd.testing.assert_frame_equal(test.metadata, original_test.metadata)
        self.assertEqual(train.y.mean(), 0.5)
        changed = self.rows.copy()
        changed.loc[changed.task_id == "task-0", "success"] = 0
        train, test = self.split(dataset(changed), divergent_tasks_only=True)
        self.assertTrue(set(train.metadata.task_id) <= set(original_train.metadata.task_id))
        self.assertTrue(set(test.metadata.task_id) <= set(original_test.metadata.task_id))
        self.assertNotIn("task-0", set(train.metadata.task_id) | set(test.metadata.task_id))

    def test_training_pipeline_reuses_manifest_and_turn_cutoffs(self):
        result = train_and_evaluate(LinearProbe(), dataset=self.data, manifest=self.manifest,
                                    train_max_turn=1, eval_max_turn=2)
        self.assertEqual((result["train_samples"], result["eval_samples"]), (8, 16))
        self.assertTrue(np.isfinite(result["eval_metrics"]["accuracy"]))
        train, test = self.split()
        result = train_and_evaluate(LinearProbe(), train_set=train, eval_set=test, max_turn=1)
        self.assertEqual((result["train_samples"], result["eval_samples"]), (8, 8))
        with self.assertRaisesRegex(ValueError, "pre-split"):
            train_and_evaluate(LinearProbe(), train_set=train, eval_set=test, manifest=self.manifest)

    def test_existing_random_split_remains_available(self):
        train, test = split_activation_dataset(self.data, eval_size=0.25)
        self.assertEqual((train.num_samples, test.num_samples), (24, 8))
        self.assertTrue(set(train.metadata.task_id).isdisjoint(test.metadata.task_id))


if __name__ == "__main__":
    unittest.main()
