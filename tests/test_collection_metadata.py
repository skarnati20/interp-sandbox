"""Adapter provenance and model-independent split assignments during collection."""

from dataclasses import replace
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
import torch

from src.activations import ExtractionResult
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.benchmark.huggingface import load_source_dataset
from src.benchmark.textcraft import TextCraftBenchmark
from src.data import (apply_activation_manifest, create_manifest_from_activations,
                      load_activation_dataset, split_activation_dataset)
from src.runner import Runner
from src.splits import create_split_manifest, load_split_manifest, save_split_manifest
from src.store import ShardWriter


class FixtureBenchmark(BaseBenchmark):
    source_namespace = "fixture-benchmark"
    source_provenance = {"kind": "generated", "generator": "fixture", "generator_revision": "code-sha"}

    def __init__(self):
        self.tasks = [TaskInstance(str(i), "Do the task", "Be helpful", info={"task_idx": i}) for i in range(4)]

    def list_tasks(self, split=None):
        return self.select_source_tasks(self.tasks, split)

    def reset(self, task, seed=None):
        return StepObservation("Ready")

    def step(self, action_str):
        return StepObservation("Complete", 1.0, True)

    def evaluate(self, task):
        return True

    def close(self):
        pass


class FixtureExtractor:
    num_layers = 4

    def __init__(self, model_id="fixture/model-a"):
        self.metadata = {"model_id": model_id, "model_revision": "model-commit", "tokenizer_revision": "tokenizer-commit"}
        self.generation_defaults = {"top_p": 0.95, "temperature": 1.0}

    def step_forward(self, messages, temperature=0.0, max_new_tokens=512):
        return ExtractionResult("done", torch.ones(4, 3, 2, dtype=torch.float16), 2, 1)


class TestAdapterProvenance(unittest.TestCase):
    @patch("huggingface_hub.HfApi")
    @patch("datasets.load_dataset")
    def test_official_names_and_actual_dataset_revision(self, load, api):
        api.return_value.dataset_info.return_value = SimpleNamespace(sha="resolved-dataset-commit")
        partition = Dataset.from_dict({"id": ["x"]})
        partition.info.config_name = "configuration"
        load.return_value = DatasetDict({"evaluation.v1": partition})
        partitions, provenance = load_source_dataset("owner/tasks", config="configuration", revision="branch")
        self.assertEqual(list(partitions), ["evaluation.v1"])
        self.assertEqual(provenance["revision"], "resolved-dataset-commit")
        load.assert_called_once_with("owner/tasks", name="configuration", revision="resolved-dataset-commit", token=None)

    @patch("huggingface_hub.HfApi")
    @patch("datasets.load_dataset")
    def test_explicit_release_is_selected_at_load_boundary(self, load, api):
        api.return_value.dataset_info.return_value = SimpleNamespace(sha="resolved-commit")
        partition = Dataset.from_dict({"task_id": ["BigCodeBench/0"]})
        partition.info.config_name = "default"
        load.return_value = partition
        partitions, provenance = load_source_dataset("bigcode/bigcodebench", source_split="v0.1.4")
        self.assertEqual(list(partitions), ["v0.1.4"])
        self.assertEqual(len(partitions["v0.1.4"]), 1)
        self.assertEqual(provenance["revision"], "resolved-commit")
        load.assert_called_once_with("bigcode/bigcodebench", name=None, revision="resolved-commit",
                                     token=None, split="v0.1.4")

    def test_textcraft_goal_groups_and_generated_provenance(self):
        benchmark = TextCraftBenchmark(num_tasks=4)
        task = benchmark.list_tasks()[0]
        other = replace(task, task_id=task.task_id + "-another-instance")
        self.assertEqual(benchmark.task_record(task, "crafting_goal").group_id,
                         benchmark.task_record(other, "crafting_goal").group_id)
        self.assertNotEqual(benchmark.task_record(task).task_uid, benchmark.task_record(other).task_uid)
        self.assertEqual(benchmark.source_provenance["kind"], "generated")
        self.assertEqual(len(benchmark.source_provenance["recipes_revision"]), 64)
        self.assertIsNone(task.source_split)
        with self.assertRaises(ValueError):
            benchmark.list_tasks("train")
        with self.assertRaises(ValueError):
            benchmark.get_group_key(task, "unsupported")


class TestCollectionMetadata(unittest.TestCase):
    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in Path(directory).rglob("*") if path.is_file()}

    def test_manifest_reused_across_models_and_incomplete_collections(self):
        benchmark = FixtureBenchmark()
        manifest = create_split_manifest(benchmark.task_catalog(), policy="group_holdout", test_size=0.5)
        with tempfile.TemporaryDirectory() as directory:
            for model in ("model-a", "model-b"):
                output = Path(directory) / model
                runner = Runner(FixtureExtractor(model), benchmark, output_dir=output,
                                layer_ids=[1, 3], split_manifest=manifest, verbose=False)
                runner.run_benchmark(max_tasks=1 if model == "model-a" else 4, max_steps=1)
                rows = load_activation_dataset(output, cache_dir=Path(directory) / "cache")
                for row in rows:
                    entry = manifest["tasks"][row["task_uid"]]
                    self.assertEqual(row["probe_split"], entry["probe_split"])
                    self.assertEqual(row["layer_ids"], [1, 3])
                    self.assertEqual(row["model_revision"], "model-commit")
                    self.assertIsNone(row["source_split"])
                splits = split_activation_dataset(rows, manifest)
                self.assertEqual(sum(len(partition) for partition in splits.values()), len(rows))
                metadata = json.loads((output / "run_metadata.json").read_text())
                self.assertEqual(metadata["layer_ids"], [1, 3])
                self.assertEqual(metadata["manifest_id"], manifest["manifest_id"])
                self.assertTrue((output / "split_manifest.json").exists())

    def test_invalid_model_layers_fail_before_collection(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory, layer_ids=[4])

    def test_run_metadata_records_selected_tasks_and_effective_generation(self):
        benchmark = FixtureBenchmark()
        benchmark.source_provenance = {"kind": "huggingface", "dataset_id": "fixture/tasks", "revision": "commit"}
        benchmark.tasks = [replace(task, source_split="evaluation.v1" if i in (1, 2) else "other")
                           for i, task in enumerate(benchmark.tasks)]
        extractor = FixtureExtractor()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            extractor, "step_forward", wraps=extractor.step_forward
        ) as forward:
            runner = Runner(extractor, benchmark, output_dir=directory, verbose=False)
            self.assertFalse((Path(directory) / "run_metadata.json").exists())
            runner.run_benchmark(split="evaluation.v1", max_tasks=1, max_steps=3,
                                 n_rollouts=2, temperature=0.0, max_new_tokens=17)
            metadata = json.loads((Path(directory) / "run_metadata.json").read_text())
            self.assertEqual(metadata["n_rollouts"], 2)
            self.assertEqual(metadata["max_steps"], 3)
            self.assertEqual(metadata["generation"]["requested_temperature"], 0.0)
            self.assertEqual(metadata["generation"]["effective_temperature"], 0.7)
            self.assertEqual(metadata["generation"]["max_new_tokens"], 17)
            self.assertEqual(metadata["generation"]["model_defaults"]["top_p"], 0.95)
            self.assertTrue(metadata["generation"]["do_sample"])
            self.assertEqual(metadata["task_selection"]["source_split"], "evaluation.v1")
            self.assertEqual(metadata["task_selection"]["max_tasks"], 1)
            self.assertEqual(metadata["task_selection"]["selected_task_count"], 1)
            self.assertEqual(metadata["task_selection"]["tasks"][0]["task_id"], "1")
            self.assertEqual(metadata["task_selection"]["tasks"][0]["task_idx"], 1)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["completed_episodes"], 2)
            self.assertEqual(len(forward.call_args_list), 2)
            for call in forward.call_args_list:
                self.assertEqual(call.kwargs, {"temperature": 0.7, "max_new_tokens": 17})

    def test_existing_collection_is_rejected_without_touching_initial_or_later_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory, shard_size=1, verbose=False)
            runner.run_benchmark(max_steps=1)
            before = self.snapshot(directory)
            self.assertIn("features/feat.shard0.npz", before)
            self.assertIn("features/feat.shard7.npz", before)
            with self.assertRaisesRegex(FileExistsError, "Resume is not supported"):
                Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory, shard_size=1, verbose=False)
            self.assertEqual(self.snapshot(directory), before)
            with self.assertRaisesRegex(RuntimeError, "already started"):
                runner.run_benchmark(max_tasks=1, max_steps=1)
            self.assertEqual(self.snapshot(directory), before)

    def test_orphan_outputs_and_two_prepared_runners_are_protected(self):
        for filename in ("run_metadata.json", "features/feat.shard9.npz", "run_summary.parquet", "split_manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"existing output")
                before = self.snapshot(directory)
                with self.assertRaises(FileExistsError):
                    Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory)
                self.assertEqual(before, self.snapshot(directory))
        with tempfile.TemporaryDirectory() as directory:
            first = Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory, verbose=False)
            second = Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory, verbose=False)
            first.run_benchmark(max_tasks=1, max_steps=1)
            before = self.snapshot(directory)
            with self.assertRaises(FileExistsError):
                second.run_benchmark(max_tasks=1, max_steps=1)
            self.assertEqual(before, self.snapshot(directory))

    def test_shard_writer_rejects_existing_and_late_conflicting_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ShardWriter(directory, shard_size=1)
            meta_path = Path(directory) / "meta.shard0.jsonl"
            meta_path.write_bytes(b"keep me")
            with self.assertRaises(FileExistsError):
                writer.add_step(torch.ones(1, 2), {"task_id": "a"})
            self.assertEqual(meta_path.read_bytes(), b"keep me")
            self.assertFalse((Path(directory) / "feat.shard0.npz").exists())
            with self.assertRaises(FileExistsError):
                ShardWriter(directory)

    def test_failed_collection_preserves_completed_episodes_and_planned_metadata(self):
        extractor = FixtureExtractor()
        first_result = extractor.step_forward([])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            extractor, "step_forward", side_effect=[first_result, RuntimeError("fixture failure")]
        ):
            runner = Runner(extractor, FixtureBenchmark(), output_dir=directory, verbose=False)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                runner.run_benchmark(max_steps=1, max_new_tokens=23)
            metadata = json.loads((Path(directory) / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_episodes"], 1)
            self.assertEqual(metadata["task_selection"]["selected_task_count"], 4)
            self.assertEqual(metadata["generation"]["max_new_tokens"], 23)
            rows = load_activation_dataset(directory, cache_dir=Path(directory) / "cache")
            self.assertEqual(len(rows), 2)  # Both anchors of the one completed episode.
            self.assertEqual(set(rows["task_id"]), {"0"})
            with self.assertRaises(FileExistsError):
                Runner(FixtureExtractor(), FixtureBenchmark(), output_dir=directory)

    def test_single_task_api_also_claims_and_finalizes_output(self):
        benchmark = FixtureBenchmark()
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(FixtureExtractor(), benchmark, output_dir=directory, verbose=False)
            result = runner.run_task(benchmark.tasks[2], max_steps=1, max_new_tokens=19, rollout_k=3)
            self.assertEqual(result["task_id"], "2")
            metadata = json.loads((Path(directory) / "run_metadata.json").read_text())
            self.assertEqual(metadata["mode"], "single_task")
            self.assertEqual(metadata["rollout_k"], 3)
            self.assertEqual(metadata["task_selection"]["tasks"][0]["task_id"], "2")
            self.assertEqual(metadata["generation"]["max_new_tokens"], 19)
            with self.assertRaises(RuntimeError):
                runner.run_task(benchmark.tasks[1])

    def test_collectors_reject_occupied_outputs_before_model_or_benchmark_loading(self):
        benchmark_names = {
            "collect_bigcodebench.py": "BigCodeBenchmark", "collect_textcraft.py": "TextCraftBenchmark",
            "collect_miniwob.py": "MiniWoBBenchmark",
        }
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run_metadata.json").write_text("existing")
            for filename, benchmark_name in benchmark_names.items():
                spec = importlib.util.spec_from_file_location(Path(filename).stem, scripts / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with self.subTest(script=filename), patch("sys.argv", [filename, "--output_dir", directory]), \
                        patch.object(module, "ActivationExtractor") as extractor, \
                        patch.object(module, benchmark_name) as benchmark:
                    with self.assertRaises(FileExistsError):
                        module.main()
                    extractor.assert_not_called()
                    benchmark.assert_not_called()
            self.assertEqual((Path(directory) / "run_metadata.json").read_text(), "existing")


class TestCollectFirstGrouping(unittest.TestCase):
    def test_textcraft_collection_without_manifest_can_be_explicitly_regrouped(self):
        benchmark = TextCraftBenchmark(num_tasks=4)
        # Two distinct task instances with the same crafting goal.
        benchmark.task_list.append(replace(benchmark.task_list[0], task_id="textcraft/repeated-goal"))
        catalog = benchmark.task_catalog("crafting_goal")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "collect-first"
            runner = Runner(FixtureExtractor(), benchmark, output_dir=output, verbose=False)
            runner.run_benchmark(max_steps=1)
            rows = load_activation_dataset(output, cache_dir=Path(directory) / "cache")
            old_groups = list(rows["group_id"])
            self.assertEqual(set(rows["grouping"]), {"task"})
            self.assertTrue(all(value is None for value in rows["probe_split"]))
            with self.assertRaisesRegex(ValueError, "Group differs"):
                create_manifest_from_activations(rows, policy="group_holdout", task_catalog=catalog,
                                                 grouping="crafting_goal")
            manifest = create_manifest_from_activations(
                rows, policy="group_holdout", task_catalog=catalog, grouping="crafting_goal", allow_regroup=True)
            changed_source = [replace(record, provenance={**record.provenance, "recipes_revision": "changed"})
                              for record in catalog]
            with self.assertRaisesRegex(ValueError, "Source provenance differs"):
                create_manifest_from_activations(rows, policy="group_holdout", task_catalog=changed_source,
                                                 grouping="crafting_goal", allow_regroup=True)
            path = Path(directory) / "goal-manifest.json"
            save_split_manifest(manifest, path)
            manifest = load_split_manifest(path)
            annotated = apply_activation_manifest(rows, manifest, task_catalog=catalog, allow_regroup=True)
            self.assertEqual(annotated[0]["group_id"], annotated[-1]["group_id"])
            self.assertEqual(annotated[0]["probe_split"], annotated[-1]["probe_split"])
            self.assertEqual(set(annotated["grouping"]), {"crafting_goal"})
            self.assertEqual(list(rows["group_id"]), old_groups)  # The original collection is unchanged.
            partial = split_activation_dataset(rows.select([0]), manifest, task_catalog=catalog, allow_regroup=True)
            self.assertEqual(sum(len(partition) for partition in partial.values()), 1)
            for column in ("probe_split", "manifest_id"):
                marked = rows.remove_columns(column).add_column(column, ["established"] + [None] * (len(rows) - 1))
                with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Cannot regroup"):
                    create_manifest_from_activations(marked, policy="group_holdout", task_catalog=catalog,
                                                     grouping="crafting_goal", allow_regroup=True)
            with self.assertRaisesRegex(ValueError, "Cannot regroup"):
                apply_activation_manifest(annotated, manifest, task_catalog=catalog, allow_regroup=True)


if __name__ == "__main__":
    unittest.main()
