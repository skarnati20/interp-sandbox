"""
Unit and integration tests for the BigCodeBench benchmark adapter.
"""

from pathlib import Path
import json
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.benchmark.bigcodebench.bigcodebench import BigCodeBenchmark


class TestBigCodeBenchCodeExtraction(unittest.TestCase):
    def test_extract_markdown_python(self):
        text = (
            "Here is the solution to your problem:\n\n"
            "```python\n"
            "import math\n\n"
            "def task_func(x):\n"
            "    return math.sqrt(x)\n"
            "```\n\n"
            "This computes the square root."
        )
        code = BigCodeBenchmark.extract_code(text, entry_point="task_func")
        self.assertIn("def task_func(x):", code)
        self.assertIn("import math", code)
        self.assertNotIn("Here is the solution", code)
        self.assertNotIn("This computes", code)

    def test_strip_think_tags(self):
        text = (
            "<think>\n"
            "First I need to import itertools and shuffle.\n"
            "Then iterate over pairs.\n"
            "</think>\n"
            "```python\n"
            "def task_func(numbers):\n"
            "    return sum(numbers)\n"
            "```"
        )
        code = BigCodeBenchmark.extract_code(text, entry_point="task_func")
        self.assertNotIn("<think>", code)
        self.assertNotIn("First I need to", code)
        self.assertEqual(code, "def task_func(numbers):\n    return sum(numbers)")

    def test_extract_multiple_blocks_prefers_entry_point(self):
        text = (
            "Here is how you call it:\n"
            "```python\n"
            "print('example')\n"
            "```\n"
            "And here is the actual implementation:\n"
            "```python\n"
            "def task_func(val):\n"
            "    return val * 2\n"
            "```"
        )
        code = BigCodeBenchmark.extract_code(text, entry_point="task_func")
        self.assertIn("def task_func(val):", code)
        self.assertNotIn("print('example')", code)

    def test_raw_code_extraction(self):
        text = "import os\n\ndef task_func():\n    return os.getcwd()\n"
        code = BigCodeBenchmark.extract_code(text, entry_point="task_func")
        self.assertEqual(code, "import os\n\ndef task_func():\n    return os.getcwd()")

    def test_prepend_code_prompt_when_header_missing(self):
        body = "    return sum(numbers)\n"
        prompt = "def task_func(numbers):\n"
        code = BigCodeBenchmark.extract_code(body, entry_point="task_func", code_prompt=prompt)
        self.assertIn("def task_func(numbers):", code)
        self.assertIn("return sum(numbers)", code)


class TestBigCodeBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise real code execution without downloading a mutable benchmark.
        fixture = json.loads((Path(__file__).parent / "fixtures" / "bigcodebench_releases.json").read_text())
        cls.release_partitions = DatasetDict({name: Dataset.from_list(rows)
                                              for name, rows in fixture["partitions"].items()})
        cls.source_patch = patch("src.benchmark.bigcodebench.bigcodebench.load_source_dataset", return_value=(
            cls.release_partitions,
            {"kind": "huggingface", "dataset_id": "fixture/bigcodebench", "config": fixture["config"], "revision": "fixed-commit"},
        ))
        cls.source_patch.start()
        cls.addClassCleanup(cls.source_patch.stop)

    def setUp(self):
        self.bench = BigCodeBenchmark(subset="hard", num_tasks=5, timeout=5.0)
        self.tasks = self.bench.list_tasks()

    def tearDown(self):
        self.bench.close()

    def test_task_list(self):
        self.assertEqual(len(self.tasks), 2)
        for t in self.tasks:
            self.assertTrue(t.task_id.startswith("bigcodebench/"))
            self.assertIn("entry_point", t.info)
            self.assertEqual(t.source_split, "v0.1.4")

    def test_official_partition_and_catalog_survive_collection_limits(self):
        small = BigCodeBenchmark(num_tasks=1)
        self.assertEqual(len(small.list_tasks("v0.1.4")), 1)
        self.assertEqual(len(small.task_catalog()), 2)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            small.list_tasks("train")

    def test_default_selects_one_release_from_real_hub_layout(self):
        self.assertEqual(list(self.release_partitions), ["v0.1.0_hf", "v0.1.1", "v0.1.2", "v0.1.3", "v0.1.4"])
        self.assertEqual(len(self.bench.task_catalog()), 2)  # Not ten rows from five releases.
        self.assertEqual(len({record.task_uid for record in self.bench.task_catalog()}), 2)
        self.assertEqual({task.source_split for task in self.tasks}, {"v0.1.4"})
        self.assertEqual(self.bench.source_provenance["release"], "v0.1.4")
        self.assertTrue(all("[v0.1.4]" in task.instruction for task in self.tasks))
        self.assertEqual(self.bench.task_specs[0].libs, ["math"])

    def test_explicit_release_is_independent_of_prompt_format(self):
        selected = BigCodeBenchmark(release="v0.1.2", split="instruct")
        self.assertEqual({record.source_split for record in selected.task_catalog()}, {"v0.1.2"})
        self.assertTrue(all("[v0.1.2]" in task.instruction for task in selected.list_tasks()))
        self.assertEqual(selected.collection_settings()["prompt_format"], "instruct")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            selected.list_tasks("v0.1.4")

    def test_missing_or_empty_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            BigCodeBenchmark(release="missing")
        for release in (None, ""):
            with self.subTest(release=release), self.assertRaises(ValueError):
                BigCodeBenchmark(release=release)

    def test_reset_instruct(self):
        task = self.tasks[0]
        obs = self.bench.reset(task)
        self.assertIn("Specification:", obs.observation_text)
        self.assertIn(task.info["entry_point"], obs.observation_text)
        self.assertEqual(obs.step_reward, 0.0)
        self.assertFalse(obs.is_done)

    def test_reset_complete(self):
        bench_comp = BigCodeBenchmark(subset="hard", split="complete", num_tasks=1)
        tasks_comp = bench_comp.list_tasks()
        obs = bench_comp.reset(tasks_comp[0])
        self.assertIn("Complete the following Python function:", obs.observation_text)
        bench_comp.close()

    def test_canonical_solution_success(self):
        task = self.tasks[0]
        spec = self.bench.task_specs[task.info["task_idx"]]
        self.bench.reset(task)

        # Submit canonical solution wrapped in markdown block
        full_code = f"```python\n{spec.complete_prompt}\n{spec.canonical_solution}\n```"
        obs = self.bench.step(full_code)

        self.assertTrue(obs.is_done)
        self.assertEqual(obs.step_reward, 1.0)
        self.assertTrue(obs.info["success"])
        self.assertTrue(self.bench.evaluate(task))

    def test_failing_solution_returns_error(self):
        task = self.tasks[0]
        self.bench.reset(task)

        bad_code = "```python\ndef task_func(*args, **kwargs):\n    return -999999\n```"
        obs = self.bench.step(bad_code)

        self.assertFalse(obs.is_done)
        self.assertEqual(obs.step_reward, 0.0)
        self.assertFalse(obs.info["success"])
        self.assertIn("Test execution failed:", obs.observation_text)
        self.assertFalse(self.bench.evaluate(task))

    def test_syntax_error_handled_gracefully(self):
        task = self.tasks[0]
        self.bench.reset(task)

        broken_code = "```python\ndef task_func(:::\n```"
        obs = self.bench.step(broken_code)

        self.assertEqual(obs.step_reward, 0.0)
        self.assertIn("SyntaxError", obs.observation_text)
        self.assertFalse(self.bench.evaluate(task))

    def test_timeout_handled(self):
        bench_quick = BigCodeBenchmark(subset="hard", num_tasks=1, timeout=1.0)
        task = bench_quick.list_tasks()[0]
        bench_quick.reset(task)

        hang_code = "```python\ndef task_func(*args, **kwargs):\n    while True:\n        pass\n```"
        obs = bench_quick.step(hang_code)

        self.assertFalse(obs.is_done)
        self.assertEqual(obs.step_reward, 0.0)
        self.assertIn("TimeoutExpired", obs.observation_text)
        self.assertFalse(bench_quick.evaluate(task))
        bench_quick.close()

    def test_multi_turn_interactive_recovery(self):
        bench_interactive = BigCodeBenchmark(
            subset="hard", num_tasks=1, timeout=5.0
        )
        task = bench_interactive.list_tasks()[0]
        spec = bench_interactive.task_specs[task.info["task_idx"]]
        bench_interactive.reset(task)

        # Turn 1: Wrong solution -> not done, gets error feedback
        bad_code = "```python\ndef task_func(*args, **kwargs):\n    return None\n```"
        obs1 = bench_interactive.step(bad_code)
        self.assertFalse(obs1.is_done)
        self.assertEqual(obs1.step_reward, 0.0)

        # Turn 2: Correct solution -> done, reward 1.0
        good_code = f"```python\n{spec.complete_prompt}\n{spec.canonical_solution}\n```"
        obs2 = bench_interactive.step(good_code)
        self.assertTrue(obs2.is_done)
        self.assertEqual(obs2.step_reward, 1.0)
        self.assertTrue(bench_interactive.evaluate(task))
        bench_interactive.close()


if __name__ == "__main__":
    unittest.main()
