"""
Unit tests for the InterCode-Bash Benchmark environment.
"""

from pathlib import Path
import unittest

from src.benchmark.intercode.bash import InterCodeBashBenchmark


class TestInterCodeBashBenchmark(unittest.TestCase):
    def setUp(self):
        self.bench = InterCodeBashBenchmark()
        self.tasks = self.bench.list_tasks()

    def tearDown(self):
        self.bench.close()

    def test_task_suite_generation(self):
        self.assertEqual(len(self.tasks), 50)
        self.assertTrue(self.tasks[0].task_id.startswith("intercode-bash/"))

    def test_extract_emails_task(self):
        task = self.tasks[0]  # extract_emails
        obs = self.bench.reset(task)
        self.assertIn("data.txt", obs.observation_text)

        # Run grep command
        step_obs = self.bench.step(
            "grep -o -E '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}' data.txt | sort > emails.txt"
        )
        self.assertIn("[Return Code: 0]", step_obs.observation_text)

        # Submit task
        submit_obs = self.bench.step("submit")
        self.assertTrue(submit_obs.is_done)
        self.assertTrue(submit_obs.info["success"])
        self.assertTrue(self.bench.evaluate(task))

    def test_count_errors_task(self):
        task = self.tasks[1]  # count_errors
        obs = self.bench.reset(task)
        self.assertIn("server.log", obs.observation_text)

        # Run count command
        self.bench.step("grep -c 'ERROR' server.log > error_count.txt")

        # Evaluate
        self.assertTrue(self.bench.evaluate(task))

    def test_error_capture(self):
        task = self.tasks[0]
        self.bench.reset(task)

        # Run invalid command
        step_obs = self.bench.step("cat nonexistent_file_xyz.txt")
        self.assertIn("[Return Code:", step_obs.observation_text)
        self.assertIn("stderr:", step_obs.observation_text)
        self.assertFalse(self.bench.evaluate(task))


if __name__ == "__main__":
    unittest.main()
