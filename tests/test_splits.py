"""Source partitions, grouping conflicts, and persistent manifest reuse."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.splits import (TaskRecord, apply_split_manifest, content_hash, create_split_manifest,
                        load_split_manifest, namespaced_id, save_split_manifest, validate_split_manifest)


class TestSplitManifests(unittest.TestCase):
    def records(self, source_split="test", namespace="hf/benchmark:config"):
        return [TaskRecord(namespace, str(i), source_split,
                           {"kind": "huggingface", "dataset_id": "hf/benchmark", "revision": "actual-commit", "config": "config"})
                for i in range(8)]

    def test_test_only_source_remains_test_under_probe_holdout(self):
        records = self.records()
        manifest = create_split_manifest(records, policy="group_holdout", test_size=0.25)
        assigned = apply_split_manifest(records, manifest)
        self.assertEqual({row["source_split"] for row in assigned}, {"test"})
        self.assertEqual({row["probe_split"] for row in assigned}, {"train", "test"})
        self.assertTrue(all(row["source_provenance"]["revision"] == "actual-commit" for row in assigned))

    def test_missing_official_partitions_fail_and_mapping_does_not_invent_train(self):
        records = self.records(source_split="evaluation.v1")
        with self.assertRaisesRegex(ValueError, "do not exist"):
            create_split_manifest(records, policy="source_mapping", source_mapping={"train": "train"})
        with self.assertRaisesRegex(ValueError, "missing official"):
            create_split_manifest(records, policy="group_holdout", source_splits=["train"])
        manifest = create_split_manifest(records, policy="source_mapping", source_mapping={"evaluation.v1": "test"})
        self.assertEqual({row["probe_split"] for row in apply_split_manifest(records, manifest)}, {"test"})

    def test_group_source_mapping_conflicts_are_rejected(self):
        records = [replace(self.records()[0], source_split="train", group_key="same-goal"),
                   replace(self.records()[1], source_split="test", group_key="same-goal")]
        with self.assertRaisesRegex(ValueError, "Conflicting probe assignments"):
            create_split_manifest(records, policy="source_mapping", grouping="goal",
                                  source_mapping={"train": "train", "test": "test"})
        # A task duplicated in two source partitions is also one group.
        records[1] = replace(records[0], source_split="test")
        with self.assertRaisesRegex(ValueError, "Conflicting probe assignments"):
            create_split_manifest(records, policy="source_mapping", source_mapping={"train": "train", "test": "test"})

    def test_source_pool_exclusions_preserve_catalog_and_do_not_assign_excluded_rows(self):
        records = self.records() + [replace(self.records()[0], source_split="validation")]
        manifest = create_split_manifest(records, policy="group_holdout", source_splits=["test"])
        assigned = apply_split_manifest(records, manifest)
        self.assertIsNone(assigned[-1]["probe_split"])
        self.assertEqual(manifest["tasks"][records[0].task_uid]["source_splits"], ["test", "validation"])

    def test_manifest_persistence_and_incomplete_reuse(self):
        records = self.records()
        manifest = create_split_manifest(records, policy="group_holdout")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            save_split_manifest(manifest, path)
            restored = load_split_manifest(path)
            with self.assertRaises(FileExistsError):
                save_split_manifest(manifest, path)
        full = {row["task_uid"]: row for row in apply_split_manifest(records, restored)}
        partial = apply_split_manifest(records[::3], restored)
        self.assertEqual(partial, [full[record.task_uid] for record in records[::3]])
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            apply_split_manifest([replace(records[0], task_id="new")], restored)
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            apply_split_manifest([replace(records[0], provenance={"kind": "huggingface", "revision": "changed"})], restored)

    def test_task_namespaces_and_group_namespaces_do_not_collide(self):
        records = self.records(namespace="benchmark-a")[:1] + self.records(namespace="benchmark-b")[:1]
        manifest = create_split_manifest(records, policy="group_holdout", test_size=0.5)
        self.assertEqual(len(manifest["tasks"]), 2)
        self.assertNotEqual(namespaced_id("a/b", "c"), namespaced_id("a", "b/c"))

    def test_generated_provenance_and_conflicting_task_descriptions(self):
        records = [replace(record, source_split=None, provenance={
            "kind": "generated", "generator": "fixture", "generator_revision": "code-sha",
            "parameters": {"seed": 42}}) for record in self.records()]
        manifest = create_split_manifest(records, policy="group_holdout")
        self.assertTrue(all(row["source_split"] is None for row in apply_split_manifest(records, manifest)))
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            create_split_manifest([replace(records[0], source_split="train")], policy="group_holdout")
        with self.assertRaisesRegex(ValueError, "Conflicting group_id"):
            create_split_manifest(records + [replace(records[0], group_key="other")], policy="group_holdout")
        with self.assertRaisesRegex(ValueError, "Conflicting provenance"):
            create_split_manifest(records + [replace(records[0], provenance={"kind": "legacy"})], policy="group_holdout")

    def test_goal_grouping_and_manifest_conflict_validation(self):
        records = [replace(record, group_key=f"goal-{i // 2}") for i, record in enumerate(self.records())]
        manifest = create_split_manifest(records, policy="group_holdout", grouping="goal")
        assigned = apply_split_manifest(records, manifest)
        for i in range(0, len(records), 2):
            self.assertEqual(assigned[i]["probe_split"], assigned[i + 1]["probe_split"])
        manifest["tasks"][records[0].task_uid]["probe_split"] = "conflict"
        manifest["manifest_id"] = content_hash({k: v for k, v in manifest.items() if k != "manifest_id"})
        with self.assertRaisesRegex(ValueError, "Conflicting probe"):
            validate_split_manifest(manifest)

    def test_invalid_fractions_and_insufficient_groups(self):
        for size in (0, 1, -0.1, 1.1, float("nan")):
            with self.subTest(size=size), self.assertRaises(ValueError):
                create_split_manifest(self.records(), policy="group_holdout", test_size=size)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            create_split_manifest(self.records()[:1], policy="group_holdout")


if __name__ == "__main__":
    unittest.main()
