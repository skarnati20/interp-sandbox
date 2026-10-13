"""
Unit and integration tests for the BigCodeBench benchmark adapter.
"""

from pathlib import Path
import unittest

from src.benchmark.bigcodebench import (
    BigCodeBenchActionParser,
    BigCodeBenchmark,
)


class TestBigCodeBenchActionParser(unittest.TestCase):
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
        code = BigCodeBenchActionParser.extract_code(text, entry_point="task_func")
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
        code = BigCodeBenchActionParser.extract_code(text, entry_point="task_func")
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
        code = BigCodeBenchActionParser.extract_code(text, entry_point="task_func")
        self.assertIn("def task_func(val):", code)
        self.assertNotIn("print('example')", code)

    def test_raw_code_extraction(self):
        text = "import os\n\ndef task_func():\n    return os.getcwd()\n"
        code = BigCodeBenchActionParser.extract_code(text, entry_point="task_func")
        self.assertEqual(code, "import os\n\ndef task_func():\n    return os.getcwd()")

    def test_prepend_code_prompt_when_header_missing(self):
        body = "    return sum(numbers)\n"
        prompt = "def task_func(numbers):\n"
        code = BigCodeBenchActionParser.extract_code(body, entry_point="task_func", code_prompt=prompt)
        self.assertIn("def task_func(numbers):", code)
        self.assertIn("return sum(numbers)", code)


class TestBigCodeBenchmark(unittest.TestCase):
    def setUp(self):
        # Use bundled sample tasks for deterministic, offline testing
        self.bench = BigCodeBenchmark(sample_only=True, num_tasks=5, timeout=5.0)
        self.tasks = self.bench.list_tasks()

    def tearDown(self):
        self.bench.close()

    def test_task_list(self):
        self.assertEqual(len(self.tasks), 5)
        for t in self.tasks:
            self.assertTrue(t.task_id.startswith("bigcodebench/"))
            self.assertIn("entry_point", t.info)

    def test_reset_instruct(self):
        task = self.tasks[0]
        obs = self.bench.reset(task)
        self.assertIn("Specification:", obs.observation_text)
        self.assertIn(task.info["entry_point"], obs.observation_text)
        self.assertEqual(obs.step_reward, 0.0)
        self.assertFalse(obs.is_done)

    def test_reset_complete(self):
        bench_comp = BigCodeBenchmark(sample_only=True, split="complete", num_tasks=1)
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

        self.assertTrue(obs.is_done)
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
        bench_quick = BigCodeBenchmark(sample_only=True, num_tasks=1, timeout=1.0)
        task = bench_quick.list_tasks()[0]
        bench_quick.reset(task)

        hang_code = "```python\ndef task_func(*args, **kwargs):\n    while True:\n        pass\n```"
        obs = bench_quick.step(hang_code)

        self.assertTrue(obs.is_done)
        self.assertEqual(obs.step_reward, 0.0)
        self.assertIn("TimeoutExpired", obs.observation_text)
        self.assertFalse(bench_quick.evaluate(task))
        bench_quick.close()

    def test_multi_turn_interactive_recovery(self):
        bench_interactive = BigCodeBenchmark(
            sample_only=True, num_tasks=1, single_turn=False, timeout=5.0
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
