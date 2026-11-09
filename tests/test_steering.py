from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import torch

from src.benchmark.base import TaskInstance
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
        self.mock_extractor.tokenizer.decode.return_value = "generated response"

        # Mock model forward outputs
        class ModelOutput:
            def __init__(self):
                self.logits = torch.zeros((1, 1, 1000))
                self.logits[0, -1, 42] = 10.0
                self.hidden_states = (
                    torch.zeros((1, 1, 64)),
                    torch.zeros((1, 1, 64)),
                )

        # Real hook execution on mock layer
        self.mock_layer = MagicMock()

        def mock_register_forward_hook(hook_fn):
            self._active_hook = hook_fn
            handle = MagicMock()
            def remove():
                self._active_hook = None
            handle.remove = remove
            return handle

        self._active_hook = None
        self.mock_layer.register_forward_hook.side_effect = mock_register_forward_hook
        self.mock_extractor.model.model.layers = [self.mock_layer]

        def mock_model_forward(*args, **kwargs):
            out = ModelOutput()
            if self._active_hook is not None:
                # Run hook on hidden states
                res = self._active_hook(self.mock_layer, args, (out.hidden_states[1],))
                out.hidden_states = (out.hidden_states[0], res[0] if isinstance(res, tuple) else res)
            return out

        self.mock_extractor.model.side_effect = mock_model_forward

        self.probe = DummyProbe(score=0.8)
        self.steerer = CentroidSteerer(alpha=1.0)
        self.steerer.steering_direction = np.ones(64)

        self.eval_set = [
            TaskInstance(
                task_id="task_1",
                instruction="Write a hello function",
                system_prompt="You are a helpful assistant.",
            )
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_and_tasks(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            eval_set=self.eval_set,
            output_dir=self.temp_dir.name,
            layer_id=0,
            verbose=False,
        )
        tasks = runner.get_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "task_1")

    def test_train_intervention(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            eval_set=self.eval_set,
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
        # Probe score is 0.8, threshold is 0.5 -> No failure, steerer.intervene is NOT called
        self.probe.score = 0.8
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            eval_set=self.eval_set,
            output_dir=self.temp_dir.name,
            layer_id=0,
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        task = self.eval_set[0]
        res = runner.run_task(task, max_steps=2)

        self.assertEqual(res["num_tokens"], 2)
        self.assertEqual(res["steered_steps"], 0)
        self.steerer.intervene.assert_not_called()

    def test_run_task_with_intervention(self):
        # Probe score is 0.2, threshold is 0.5 -> Failure detected, steerer.intervene IS called
        self.probe.score = 0.2
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            eval_set=self.eval_set,
            output_dir=self.temp_dir.name,
            layer_id=0,
            threshold=0.5,
            verbose=False,
        )

        self.steerer.intervene = MagicMock(side_effect=self.steerer.intervene)

        task = self.eval_set[0]
        res = runner.run_task(task, max_steps=2)

        self.assertEqual(res["num_tokens"], 2)
        self.assertEqual(res["steered_steps"], 2)
        self.assertEqual(self.steerer.intervene.call_count, 2)

    def test_run_eval_tasks(self):
        runner = SteeringRunner(
            extractor=self.mock_extractor,
            probe=self.probe,
            steerer=self.steerer,
            eval_set=self.eval_set,
            output_dir=self.temp_dir.name,
            layer_id=0,
            verbose=False,
        )

        results = runner.run_eval_tasks(max_tasks=1, max_steps=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["num_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
