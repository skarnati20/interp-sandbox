from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import torch

from src.activations import ExtractionResult
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.probe.base import Probe
from src.steering.steerers.centroid import CentroidSteerer
from src.steering_runner import SteeringRunner
from src.store import ActivationDataset


class DummyProbe(Probe):
    def __init__(self, score: float = 0.8):
        self.score = score
        self.fitted = False

    def fit(self, train_set: ActivationDataset) -> None:
        self.fitted = True

    def predict(
        self,
        eval_set: np.ndarray | ActivationDataset,
        return_proba: bool = False,
        threshold: float = 0.5,
    ) -> np.ndarray:
        raw_x = eval_set.X if hasattr(eval_set, "X") else eval_set
        n = raw_x.shape[0] if hasattr(raw_x, "shape") else 1
        probs = np.full(n, self.score)
        if return_proba:
            return probs
        return (probs >= threshold).astype(int)


class DummyBenchmark(BaseBenchmark):
    def __init__(self, done_at_step: int = 2, success: bool = True):
        self.done_at_step = done_at_step
        self.success = success
        self.step_count = 0

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        return [
            TaskInstance(
                task_id="task_1",
                instruction="Craft planks.",
                system_prompt="You are an agent in Minecraft.",
                info={"max_steps": 5},
            )
        ]

    def reset(self, task: TaskInstance, seed: int | None = None) -> StepObservation:
        self.step_count = 0
        return StepObservation(observation_text="Inventory: 1 log", step_reward=0.0, is_done=False)

    def step(self, action_str: str) -> StepObservation:
        self.step_count += 1
        done = self.step_count >= self.done_at_step
        return StepObservation(
            observation_text=f"Crafted item at step {self.step_count}",
            step_reward=1.0 if done else 0.0,
            is_done=done,
        )

    def evaluate(self, task: TaskInstance) -> bool:
        return self.success

    def close(self) -> None:
        pass


class TestCentroidSteerer(unittest.TestCase):
    def setUp(self):
        x_pos = np.array([[9.0, 10.0], [11.0, 10.0]])
        x_neg = np.array([[1.0, 2.0], [3.0, 2.0]])
        x = np.vstack([x_pos, x_neg])
        y = np.array([1, 1, 0, 0])

        self.dataset = ActivationDataset(
            X=x,
            y=y,
            layer_ids=[20],
            metadata=pd.DataFrame({"id": range(4)}),
        )

    def test_fit_and_direction(self):
        steerer = CentroidSteerer(alpha=1.0)
        steerer.fit(self.dataset)

        expected_dir = np.array([8.0, 8.0])
        np.testing.assert_allclose(steerer.steering_direction, expected_dir)
        np.testing.assert_allclose(steerer.get_vector(), expected_dir)

    def test_alpha_scaling(self):
        steerer = CentroidSteerer(alpha=2.5)
        steerer.fit(self.dataset)

        expected_vec = 2.5 * np.array([8.0, 8.0])
        np.testing.assert_allclose(steerer.get_vector(), expected_vec)

    def test_normalize(self):
        steerer = CentroidSteerer(alpha=1.0, normalize=True)
        steerer.fit(self.dataset)

        expected_dir = np.array([8.0, 8.0]) / np.linalg.norm([8.0, 8.0])
        np.testing.assert_allclose(steerer.steering_direction, expected_dir)

    def test_intervene(self):
        steerer = CentroidSteerer(alpha=0.5)
        steerer.fit(self.dataset)

        act = np.array([1.0, 1.0])
        steered_act = steerer.intervene(act)
        np.testing.assert_allclose(steered_act, np.array([5.0, 5.0]))

    def test_validation_errors(self):
        steerer = CentroidSteerer()
        with self.assertRaises(ValueError):
            steerer.intervene(np.array([1.0, 1.0]))

        all_ones = ActivationDataset(
            X=np.ones((4, 2)),
            y=np.ones(4),
            layer_ids=[20],
            metadata=pd.DataFrame({"id": range(4)}),
        )
        with self.assertRaises(ValueError):
            steerer.fit(all_ones)


class TestSteeringRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.mock_extractor = MagicMock()
        self.mock_extractor.device = "cpu"

        # Mock tokenizer
        self.mock_extractor.tokenizer.apply_chat_template.return_value = "Prompt text"
        self.mock_extractor.tokenizer.return_value = MagicMock(input_ids=torch.tensor([[10, 20]]))
        self.mock_extractor.tokenizer.eos_token_id = 999
        self.mock_extractor.tokenizer.decode.return_value = "craft 1 planks"

        # Mock step_forward
        dummy_hidden = torch.zeros((2, 5, 64), dtype=torch.float16)
        self.mock_result = ExtractionResult(
            completion_text="craft 1 planks",
            hidden_states=dummy_hidden,
            prompt_tokens=2,
            completion_tokens=3,
        )
        self.mock_extractor.step_forward.return_value = self.mock_result

        # Mock model for pre_gen forward pass
        class ModelOutput:
            def __init__(self):
                self.hidden_states = (
                    torch.zeros((1, 2, 64)),
                    torch.zeros((1, 2, 64)),
                    torch.zeros((1, 2, 64)),
                )

        self.mock_extractor.model.return_value = ModelOutput()

        # Mock layer for hook registration
        self.mock_layer = MagicMock()
        self.mock_hook_handle = MagicMock()
        self.mock_layer.register_forward_hook.return_value = self.mock_hook_handle
        self.mock_extractor.model.model.layers = [self.mock_layer]

        self.probe = DummyProbe(score=0.8)
        self.steerer = CentroidSteerer(alpha=1.0)
        self.steerer.steering_direction = np.ones(64)
        self.benchmark = DummyBenchmark(done_at_step=2, success=True)
        self.test_task = self.benchmark.list_tasks()[0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_and_tasks(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            verbose=False,
        )
        tasks = runner.get_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "task_1")

        custom_task = TaskInstance(task_id="custom", instruction="do it", system_prompt="prompt")
        custom_tasks = runner.get_tasks(tasks=[custom_task])
        self.assertEqual(len(custom_tasks), 1)
        self.assertEqual(custom_tasks[0].task_id, "custom")

    def test_train_intervention(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            verbose=False,
        )

        dataset = ActivationDataset(
            X=np.vstack([np.ones((2, 64)), np.zeros((2, 64))]),
            y=np.array([1, 1, 0, 0]),
            layer_ids=[0],
            metadata=pd.DataFrame({"id": range(4)}),
        )

        runner.train_intervention(dataset)
        self.assertTrue(self.probe.fitted)
        self.assertIsNotNone(self.steerer.steering_direction)

    def test_run_task_no_intervention(self):
        self.probe.score = 0.8
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            anchor="post_gen",
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        res = runner.run_task(self.test_task)

        self.assertTrue(res["is_success"])
        self.assertEqual(res["steered_steps"], 0)
        self.steerer.intervene.assert_not_called()

    def test_run_task_with_intervention_post_gen(self):
        self.probe.score = 0.2
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            anchor="post_gen",
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        res = runner.run_task(self.test_task)

        self.assertTrue(res["is_success"])
        self.assertGreater(res["steered_steps"], 0)
        self.assertGreater(self.steerer.intervene.call_count, 0)
        self.mock_layer.register_forward_hook.assert_called()

    def test_run_task_with_intervention_pre_gen(self):
        self.probe.score = 0.2
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            anchor="pre_gen",
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        res = runner.run_task(self.test_task)

        self.assertTrue(res["is_success"])
        self.assertGreater(res["steered_steps"], 0)
        self.assertGreater(self.steerer.intervene.call_count, 0)

    def test_multiturn_benchmark_rollout(self):
        self.benchmark.done_at_step = 3
        self.probe.score = 0.3
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            threshold=0.5,
            verbose=False,
        )

        res = runner.run_task(self.test_task, dynamic_steps=True)

        self.assertTrue(res["is_success"])
        self.assertEqual(res["num_steps"], 3)
        self.assertEqual(res["max_steps"], 5)  # From task.info["max_steps"]
        self.assertFalse(res["hit_max_rounds"])
        self.assertGreater(res["steered_steps"], 0)

    def test_max_steer_turns(self):
        self.benchmark.done_at_step = 3
        self.probe.score = 0.2
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        res = runner.run_task(self.test_task, dynamic_steps=True, max_steer_turns=1)

        self.assertTrue(res["is_success"])
        self.assertEqual(res["num_steps"], 3)
        self.assertEqual(res["steered_steps"], 1)

    def test_resampling_mode(self):
        # Test rejection sampling: first attempt fails (0.2), second attempt passes (0.9)
        scores = [0.2, 0.9, 0.2, 0.9]
        score_iter = iter(scores)

        def mock_predict(x, return_proba=False):
            s = next(score_iter, 0.8)
            return np.array([s]) if return_proba else np.array([int(s >= 0.5)])

        self.probe.predict = MagicMock(side_effect=mock_predict)
        self.benchmark.done_at_step = 2

        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            threshold=0.5,
            method="resample",
            max_resample_attempts=5,
            verbose=False,
        )

        res = runner.run_task(self.test_task)
        self.assertTrue(res["is_success"])
        # Resampling was triggered
        self.assertGreater(res["steered_steps"], 0)
        # Vector intervention was NOT called because method="resample"
        self.mock_layer.register_forward_hook.assert_not_called()

    def test_run_benchmark(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            benchmark=self.benchmark,
            output_dir=self.temp_dir.name,
            layer_id=0,
            verbose=False,
        )

        results = runner.run_benchmark(tasks=[self.test_task], max_tasks=1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["is_success"])
        self.assertEqual(results[0]["num_steps"], 2)


if __name__ == "__main__":
    unittest.main()
